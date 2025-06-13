
import numpy as np 
import pandas as pd 
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt




def ConnectToDatabase(path):
    connection = sqlite3.connect(path)
    return connection

connection = None

# Define the path to the database
database = r'C:\Users\avazb\Desktop\BigData\Lab1\Formula 1 Race Data\Formula1.sqlite'

# Establish a connection to the SQLite database
try:
    connection = ConnectToDatabase(database)
    cursor = connection.cursor()
    
    # Fetch table names
    tables = pd.read_sql_query("SELECT * FROM sqlite_master WHERE type='table';", connection)
    print("Tables in the database:\n", tables)

    # Load data into DataFrames
    circuits = pd.read_sql_query("SELECT * FROM circuits;", connection)
    drivers = pd.read_sql_query("SELECT * FROM drivers ORDER BY surname;", connection)
    drivers_details = pd.read_sql_query("""
        SELECT * FROM drivers
        JOIN results AS R ON R.driverId = drivers.driverId
        ORDER BY forename;
    """, connection)

    # Query for driver ranks
    query = """
        SELECT drivers.forename, results.rank
        FROM drivers
        JOIN results ON results.driverId = drivers.driverId
        GROUP BY results.rank 
        ORDER BY results.rank ASC;
    """
    result_by_driver = pd.read_sql_query(query, connection)
    result_by_driver['rank'] = pd.to_numeric(result_by_driver['rank'], errors='coerce')

    # Plotting the driver ranks
    plt.figure(figsize=(12, 5))
    result_by_driver.plot(x='forename', y='rank', kind='bar', title='Driver Rank')
    plt.xlabel('Driver Forename')
    plt.ylabel('Rank')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    driver_circuits = pd.read_sql("""SELECT drivers.forename,
                                       drivers.driverId,
                                       count(circuits.circuitId) AS number_of_circuits
                                FROM drivers
                                JOIN results ON results.driverId = drivers.driverId
                                JOIN races ON races.raceId = results.raceId
                                JOIN circuits ON circuits.circuitId = races.circuitId
                                WHERE races.year =2009
                                GROUP BY drivers.forename, drivers.driverId
                                ORDER BY drivers.forename ASC;
    """, connection)

    driver_circuits.plot(x='forename', y='number_of_circuits', kind='bar', figsize=(12,5), title='Driver circuits')
    plt.xlabel('Forename')
    plt.ylabel('Numer of circuits')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    nationality_counts = drivers['nationality'].value_counts()
    nationality_df = pd.DataFrame({'Nationality': nationality_counts.index, 'Count': nationality_counts.values})

    plt.figure(figsize=(10, 8))
    sns.heatmap(nationality_df.pivot_table(index='Nationality', values='Count', aggfunc='sum'), cmap='coolwarm', annot=True, fmt='d')
    plt.title('Number of Drivers by Nationality')
    plt.xlabel('')
    plt.ylabel('Nationality')
    plt.show()

except Exception as e:
    print("An error occurred:", e)


finally:
    # Close the database connection
    if connection:
        connection.close()

