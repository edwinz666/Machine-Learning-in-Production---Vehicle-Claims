"""
This module defines the database models using SQLAlchemy.

The module uses SQLAlchemy's ORM capabilities to map Python
classes to database tables.
The structure and fields of the Freq class are configured
to match the corresponding database for insurance claims.
In addition, the module reads a csv file and outputs to
a SQLite database if one has not been substantiated yet
(in terms of the db_create_path supplied in the environment
variables)
"""

import os
import sqlite3

import pandas as pd
from loguru import logger
from sqlalchemy import INTEGER, REAL, VARCHAR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from config import db_settings


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""

    pass  # noqa: WPS420, WPS604


class Freq(Base):
    """
    SQLAlchemy model class for insurance claims data.

    Attributes:
        • IDpol: policy number (unique identifier)
        • ClaimNb: number of claims on the given policy
        • Exposure: total exposure in yearly units
        • Area: area code (categorical, ordinal)
        • VehPower: power of the car (categorical, ordinal)
        • VehAge: age of the car in years
        • DrivAge: age of the (most common) driver in years
        • BonusMalus: bonus-malus level between 50 and 230
            (with reference level 100)
        • VehBrand: car brand (categorical, nominal)
    """

    __tablename__ = db_settings.table_name

    IDpol: Mapped[str] = mapped_column(INTEGER(), primary_key=True)
    ClaimNb: Mapped[float] = mapped_column(INTEGER())
    Exposure: Mapped[int] = mapped_column(REAL())
    Area: Mapped[int] = mapped_column(VARCHAR())
    VehPower: Mapped[int] = mapped_column(INTEGER())
    VehAge: Mapped[int] = mapped_column(INTEGER())
    DrivAge: Mapped[str] = mapped_column(INTEGER())
    BonusMalus: Mapped[str] = mapped_column(INTEGER())
    VehBrand: Mapped[str] = mapped_column(VARCHAR())
    VehGas: Mapped[str] = mapped_column(VARCHAR())
    Density: Mapped[str] = mapped_column(REAL())
    Region: Mapped[str] = mapped_column(VARCHAR())


def csv_to_sqlite(
    csv_file_path: str = db_settings.csv_data_filename,
    sqlite_db_path: str = db_settings.db_create_path,
    table_name: str = db_settings.table_name,
) -> None:  # noqa: DAR101
    """
    Read in a csv and export to sqlite.

    This is used as an example of a database for use as
    part of a machine learning pipeline.
    """
    # Read CSV file into DataFrame
    df = pd.read_csv(csv_file_path)

    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(sqlite_db_path)

    # Write the DataFrame to SQLite database
    df.to_sql(
        name=table_name,
        con=conn,
        # if_table_exists='replace',  # noqa: E800
    )

    # Commit changes and close connection
    conn.commit()
    conn.close()

    logger.info(
        f'Data from {csv_file_path} has been written to '
        f'{sqlite_db_path} in table {table_name}.',
    )


# output supplied CSV to a sqlite database if the database doesn't exist
if not os.path.exists(db_settings.db_create_path):
    csv_to_sqlite()
