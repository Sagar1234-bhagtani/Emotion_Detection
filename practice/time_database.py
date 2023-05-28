import mysql.connector

# establish a connection to the database

cnx= mysql.connector.connect(
    host="localhost",
    user = "root",
    password= "root",
    database = "sagar",
    auth_plugin="mysql_native_password"
)

# create a cursor object
cursor = cnx.cursor()

# execute a query using NOW()
query = "INSERT INTO your_table (column1, column2, timestamp_column) VALUES (%s, %s, NOW())"
values = ("value1", "value2")
cursor.execute(query, values)

# commit the changes to the database
cnx.commit()

# close the cursor and connection
cursor.close()
cnx.close()