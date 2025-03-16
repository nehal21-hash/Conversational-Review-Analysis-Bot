import streamlit as st
import pandas as pd
import snowflake.connector
import google.generativeai as genai
import matplotlib.pyplot as plt
import re
import seaborn as sns

# 🔹 Google Gemini API Key
GEMINI_API_KEY = "AIzaSyCPJ1Am8Huk4QjvUzAt7fKwcJJjwObWhCc"
genai.configure(api_key=GEMINI_API_KEY)

# 🔹 Snowflake Connection Config
SNOWFLAKE_CONFIG = {
    "user": st.secrets["SNOWFLAKE_USER"],
    "password": st.secrets["SNOWFLAKE_PASSWORD"],
    "account": st.secrets["SNOWFLAKE_ACCOUNT"],
    "warehouse": st.secrets["SNOWFLAKE_WAREHOUSE"],
    "database": st.secrets["SNOWFLAKE_DATABASE"],
    "schema": st.secrets["SNOWFLAKE_SCHEMA"],
}

def connect_snowflake():
    """Establish Snowflake connection with explicit context setting"""
    try:
        conn = snowflake.connector.connect(**SNOWFLAKE_CONFIG)
        cursor = conn.cursor()
        # Set session context
        cursor.execute(f"USE DATABASE {SNOWFLAKE_CONFIG['database']};")
        cursor.execute(f"USE SCHEMA {SNOWFLAKE_CONFIG['schema']};")
        cursor.execute(f"USE WAREHOUSE {SNOWFLAKE_CONFIG['warehouse']};")
        cursor.close()
        return conn
    except Exception as e:
        st.error(f"❌ Connection failed: {str(e)}")
        return None

def get_table_columns(conn, table_name):
    """Get column names for a table"""
    try:
        cursor = conn.cursor()
        cursor.execute(f"DESC TABLE {get_qualified_name(table_name)}")
        return [row[0].upper() for row in cursor.fetchall()]
    except Exception as e:
        st.error(f"❌ Column fetch failed: {str(e)}")
        return []
    finally:
        cursor.close()

def get_qualified_name(table_name):
    """Return fully qualified table name"""
    return f"{SNOWFLAKE_CONFIG['database']}.{SNOWFLAKE_CONFIG['schema']}.{table_name}"

def create_table_if_not_exists(conn, table_name, df):
    """Create table with proper data types if not exists"""
    try:
        cursor = conn.cursor()
        df.columns = df.columns.str.upper()
        column_defs = ", ".join([f'"{col}" STRING' for col in df.columns])
        qualified_name = get_qualified_name(table_name)
        
        # Create table with explicit data types
        create_sql = f"""
            CREATE TABLE IF NOT EXISTS {qualified_name} (
                {column_defs}
            )
        """
        cursor.execute(create_sql)
        st.success(f"✅ Table {qualified_name} created/verified")
    except Exception as e:
        st.error(f"❌ Table creation failed: {str(e)}")
    finally:
        cursor.close()

def insert_data_to_snowflake(conn, table_name, df):
    """Insert DataFrame data into Snowflake table"""
    try:
        cursor = conn.cursor()
        df.columns = [col.upper() for col in df.columns]
        df = df.where(pd.notna(df), None)
        df = df.astype(str).replace("None", None)
        qualified_name = get_qualified_name(table_name)
        
        # Check table existence
        cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
        if not cursor.fetchone():
            create_table_if_not_exists(conn, table_name, df)
            
        # Verify columns
        cursor.execute(f"DESC TABLE {qualified_name}")
        snowflake_columns = [row[0].upper() for row in cursor.fetchall()]
        missing_cols = [col for col in df.columns if col not in snowflake_columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
            
        # Insert data
        placeholders = ", ".join(["%s"] * len(df.columns))
        sql = f"INSERT INTO {qualified_name} ({', '.join(df.columns)}) VALUES ({placeholders})"
        cursor.executemany(sql, df.itertuples(index=False, name=None))
        conn.commit()
        st.success("✅ Data inserted successfully!")
    except Exception as e:
        conn.rollback()
        st.error(f"❌ Data insertion failed: {str(e)}")
    finally:
        cursor.close()

def generate_sql_query(user_query, table_name, conn):
    """Generate SQL query using Gemini"""
    try:
        cursor = conn.cursor()
        qualified_name = get_qualified_name(table_name)
        cursor.execute(f"SELECT * FROM {qualified_name} LIMIT 5")
        sample_rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        sample_data = "\n".join([", ".join(map(str, row)) for row in sample_rows])
        
        prompt = f"""
        Convert this natural language query to valid Snowflake SQL:
        Table: {qualified_name}
        Columns: {', '.join(col_names)}
        Sample Data:
        {sample_data}
        
        Rules:
        - Use Snowflake SQL syntax
        - Handle data type conversions (TO_DATE/TO_NUMBER if needed)
        - Match exact column names
        - No markdown formatting
        
        Query: "{user_query}"
        """
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        sql_query = re.sub(r"```sql|```", "", response.text.strip()).strip()
        
        if not sql_query.upper().startswith("SELECT"):
            sql_query = f'SELECT * FROM {qualified_name} WHERE {sql_query};'
            
        return sql_query
    except Exception as e:
        st.error(f"❌ Query generation failed: {str(e)}")
        return f"SELECT * FROM {qualified_name} LIMIT 10"

def execute_sql_query(conn, query):
    """Execute SQL query and return DataFrame"""
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        df = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])
        return df
    except Exception as e:
        st.error(f"❌ Query execution failed: {str(e)}")
        return pd.DataFrame()
    finally:
        cursor.close()

def plot_data(df):
    """Visualize DataFrame results"""
    if df.empty:
        st.warning("⚠️ No data to visualize")
        return
    
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        num_cols = df.select_dtypes(include="number").columns
        
        if len(num_cols) >= 2:
            df.plot(kind="line", ax=ax, marker="o")
            plt.xticks(rotation=45)
        elif len(num_cols) == 1:
            sns.histplot(df[num_cols[0]], kde=True, ax=ax)
        else:
            st.write("⚠️ No numerical data to plot")
        
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Visualization failed: {str(e)}")

# Streamlit UI
st.title("📊 Smart Data Analyst Bot")
st.write("Upload CSV and ask questions in natural language!")

uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head(3))
    
    table_name = "ANALYSIS_DATA"
    conn = connect_snowflake()
    
    if conn:
        with st.spinner("Configuring database..."):
            create_table_if_not_exists(conn, table_name, df)
            insert_data_to_snowflake(conn, table_name, df)
        
        user_query = st.text_input("Ask about your data:")
        if user_query:
            with st.spinner("Generating SQL..."):
                sql_query = generate_sql_query(user_query, table_name, conn)
                st.code(sql_query, language="sql")
            
            with st.spinner("Executing query..."):
                result_df = execute_sql_query(conn, sql_query)
                st.dataframe(result_df)
                
                if not result_df.empty:
                    st.subheader("Visualization")
                    plot_data(result_df)
        
        conn.close()
