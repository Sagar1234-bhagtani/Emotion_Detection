import mysql.connector


MyDB= mysql.connector.connect(
    host="localhost",
    user = "root",
    password= "root",
    database = "sagar",
    auth_plugin="mysql_native_password"
)

MyCursor =MyDB.cursor()
MyCursor.execute("CREATE TABLE IF NOT EXISTS Images (id INTEGER NOT NULL AUTO_INCREMENT PRIMARY KEY,Photo LONGBLOB NOT NULL);")
def InsertBlob(FilePath):
     with open (FilePath, "rb") as File:
        BinaryData= File.read()
     SQLStatement = "INSERT INTO Images (Photo) VALUES (%s)"
     MyCursor.execute(SQLStatement, (BinaryData, ))
     MyDB.commit()


def RetrieveBlob(ID):
     SQLStatement2 = "SELECT Photo FROM datab WHERE id = '{0}' "
     MyCursor.execute(SQLStatement2. format (str(ID)))
     MyResult= MyCursor.fetchone () [0]
     StoreFilePath = "ImageOutputs/img{0}.jpg". format(str(ID))
     print (MyResult)
     with open(StoreFilePath, "wb") as File:
        File.write(MyResult)
        File.close()



print("1. Insert Image\n2. Read Image")
MenuInput=input()
if int (MenuInput) == 1:
    UserFilePath = input("Enter File Path: ")
    InsertBlob(UserFilePath)
elif int (MenuInput) == 2:
    UserIDChoice = input ("Enter ID:")
    RetrieveBlob(UserIDChoice)