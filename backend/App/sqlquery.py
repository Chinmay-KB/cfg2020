import sqlite3, ast

current_path= '../results.db'
def getRows():
	'''
	Returns all the constituencies with current cases and other details
	'''
	conn = sqlite3.connect(current_path) #change path here
	c = conn.cursor()
	c.execute(f'''select * from main;''')
	rows=c.fetchall()
	c.execute(f'''PRAGMA table_info(main);''')
	column_names=[t[1] for t in c.fetchall()]
	c.close()
	return rows, column_names