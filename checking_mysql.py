import mysql.connector

mydb=mysql.connector.connect(host="localhost",user="root",passwd="root",database="sagar",auth_plugin="mysql_native_password")



mycursor = mydb.cursor()wefjljsd

# Define the SQL query to fetch all records from the table
sql = "SELECT * FROM todo"sdjfkljsklj

# Execute the query
mycursor.execute(sql)wsdfkklsdafnknfsad

# Fetch all the records
result = mycursor.fetchall()ksdfnkjndsfndfsj

# Print the records
for record in result:
  print(record)
