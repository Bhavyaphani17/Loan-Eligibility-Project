import sqlite3

conn = sqlite3.connect('loan_applications.db')
cursor = conn.cursor()
cursor.execute("SELECT * FROM applications")
rows = cursor.fetchall()
for row in rows:
    print(row)
conn.close()