import mysql.connector

from constants import companies

mydb = mysql.connector.connect(
  host="localhost",
  port=3306,
  user="root",
)

cursor = mydb.cursor();

cursor.execute('USE EARNINGS');
cursor.execute('DESCRIBE income_statement');
for row in cursor.fetchall():
    print(row)

for company in companies:
    cursor.execute(f"SELECT * FROM income_statement WHERE act_symbol='{company}' LIMIT 1")
    for row in cursor.fetchall():
        print(row)
