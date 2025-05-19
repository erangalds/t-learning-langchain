"""
The below example will use a SQLite connection with the Chinook database, which is a sample database that represents a digital media store. Follow these installation steps to create Chinook.db in the same directory as this notebook. You can also download and build the database via the command line:

```bash
curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 Chinook.db

```

Afterwards, place `Chinook.db` in the same directory where this code is running.

"""

from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
# replace this with the connection details of your db
from langchain_ollama import ChatOllama

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.get_usable_table_names())
llm = ChatOllama(
    model="gemma3:27b", 
    temperature=0
)

# convert question to sql query
write_query = create_sql_query_chain(llm, db)

# Execute SQL query
execute_query = QuerySQLDatabaseTool(db=db)

# Write a output formatter
def parse_sql_output(output):
    # Parse the SQL output from the message
    lines = output.split("\n")
    return '\n'.join(lines[1:len(lines)-1])

# Run the write Query
question = "How many employees are there?"
# sql_query = write_query.invoke({"question": question})
# print("SQL Query: \n", sql_query)
# # run the chain
# formatted_sql = '\n'.join(parse_sql_output(sql_query))
# print("Formatted SQL: \n", formatted_sql)


# combined chain = write_query | execute_query
combined_chain = write_query | parse_sql_output | execute_query

quesion = "How many employees are there? Give me only the SQL query as output."
result = combined_chain.invoke({"question": question})

#print(result)
print("\n\nResult: \n", result)

question = "How many albums have we sold? Give me only the SQL query as output."
result = combined_chain.invoke({"question": question})

#print(result)
print("\n\nResult: \n", result)


