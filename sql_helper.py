# -*- coding: utf-8 -*-

import json
import yaml
import pymysql
from dbutils.pooled_db import PooledDB

from ComplexEncoder import ComplexEncoder
from DBConfig import DBConfig

from utils.logger import setup_logger

logger = setup_logger('SQLHelper')

with open('db.yml', mode='r') as f:
    result = yaml.load(f, Loader=yaml.FullLoader)
    mysqlresult = result['mysql']
    host = mysqlresult["host"]
    port = mysqlresult["port"]
    username = mysqlresult["username"]
    password = str(mysqlresult["password"])
    db = mysqlresult["db"]
    charset = mysqlresult["charset"]
    maxConnectionSize = mysqlresult["max-connection-size"]
    initConnectionSize = mysqlresult["init-connection-size"]
    maxIdleSize = mysqlresult["max-idle-size"]
    blockingIfNoConnection = mysqlresult["blocking-if-no-connection"]
    dbConfig = DBConfig(host=host, port=port, username=username, password=password, db=db, charset=charset,
                        blockingIfNoConnection=blockingIfNoConnection, maxConnectionSize=maxConnectionSize,
                        initConnectionSize=initConnectionSize, maxIdleSize=maxIdleSize)

    POOL = PooledDB(
        creator=pymysql,
        # 连接池允许的最大连接数，0和None表示不限制连接数
        maxconnections=dbConfig.getMaxConnectionSize(),
        # 初始化时，链接池中至少创建的空闲的链接，0表示不创建
        mincached=dbConfig.getInitConnectionSize(),
        # 链接池中最多闲置的链接，0和None不限制
        maxcached=dbConfig.getMaxIdleSize(),
        # 连接池中如果没有可用连接后，是否阻塞等待。True，等待；False，不等待然后报错
        blocking=dbConfig.getBlockingIfNoConnection(),
        # 一个链接最多被重复使用的次数，None表示无限制
        maxusage=None,
        # 开始会话前执行的命令列表。如：["set datestyle to ...", "set time zone ..."]
        setsession=[],
        # ping MySQL服务端，检查是否服务可用。# 如：0 = None = never, 1 = default = whenever it is requested,
        # 2 = when a cursor is created, 4 = when a query is executed, 7 = always
        ping=0,
        host=dbConfig.getHost(),
        port=dbConfig.getPort(),
        user=dbConfig.getUsername(),
        passwd=dbConfig.getPassword(),
        db=dbConfig.getDb(),
        charset=dbConfig.getCharset()
    )


def connect():
    conn = POOL.connection()
    # 创建游标
    cursor = conn.cursor(pymysql.cursors.DictCursor)
    return conn, cursor


def close(conn, cursor):
    try:
        cursor.close()
    except Exception as e:
        logger.error(f"As closing DB connection oocur exception: {e}")
    finally:
        if conn is not None:
            conn.close()

def find_by_id(table_name:str, id_value, id_field_name:str="id"):
    select_sql = "select * from %s where %s = ?"
    result = {}
    try:
        conn, cursor = connect()
        select_sql = cursor.mogrify(select_sql, (table_name, id_field_name))
        select_sql = (select_sql.replace("'" + table_name + "'", table_name)
                      .replace("'" + id_field_name + "'", id_field_name)
                      .replace(id_field_name + " = ?", id_field_name + " = %s"))
        cursor.execute(select_sql, [id_value])
        result = cursor.fetchone()
    except Exception as e:
        logger.error(f"As find the record with id_val:{id_value} from DB with sql: {select_sql} oocur exception: {e}")
    finally:
        close(conn, cursor)
        return result

def find_one(sql, args=None):
    result = {}
    try:
        conn, cursor = connect()
        if not args:
            cursor.execute(sql)
        else:
            cursor.execute(sql, args)
        result = cursor.fetchone()
    except Exception as e:
        logger.error(f"As query only one record from DB with sql: {sql} oocur exception: {e}")
    finally:
        close(conn, cursor)
        return result


def find_many(sql, args=None):
    result = []
    try:
        conn, cursor = connect()
        if not args:
            cursor.execute(sql)
        else:
            cursor.execute(sql, args)
        result = cursor.fetchall()
    except Exception as e:
        logger.error(f"As query many records from DB with sql: {sql} oocur exception: {e}")
    finally:
        close(conn, cursor)
        return result

def delete_by_id(table_name:str, id_value, id_field_name:str="id"):
    delete_sql = "delete from %s where %s = ?"
    effect_row = 0
    try:
        conn, cursor = connect()
        delete_sql = cursor.mogrify(delete_sql, (table_name, id_field_name))
        delete_sql = (delete_sql.replace("'" + table_name + "'", table_name)
                      .replace("'" + id_field_name + "'", id_field_name)
                      .replace(id_field_name + " = ?", id_field_name + " = %s"))
        cursor.execute(delete_sql, [id_value])
        effect_row = cursor.execute()
        conn.commit()
    except Exception as e:
        logger.error(f"As delete the record with id_val:{id_value} from DB with sql: {delete_sql} oocur exception: {e}")
        conn.rollback()
    finally:
        close(conn, cursor)
        return effect_row

# SQL占位符为%s
def insert(sql, return_id:bool=False, args=None):
    effect_row = 0
    new_id = None
    try:
        conn, cursor = connect()
        if not args:
            effect_row = cursor.execute(sql)
        else:
            effect_row = cursor.execute(sql, args)
        conn.commit()
        if return_id:
            new_id = cursor.lastrowid
    except Exception as e:
        logger.error(f"As insert record into DB with sql: {sql} oocur exception: {e}")
        conn.rollback()
    finally:
        close(conn, cursor)
        if return_id:
            return new_id
        return effect_row

def batch_insert(sql, args=None):
    effect_row = 0
    try:
        conn, cursor = connect()
        if not args:
            effect_row = cursor.executemany(sql)
        else:
            effect_row = cursor.executemany(sql, args)
        conn.commit()
    except Exception as e:
        logger.error(f"As batch insert records into DB with sql: {sql} oocur exception: {e}")
        conn.rollback()
    finally:
        close(conn, cursor)
        return effect_row

def delete(sql, args=None):
    return insert(sql, args)


def update(sql, args=None):
    return insert(sql, args)


table_name = "sys_user"
id_field_name = "user_id"
id_value = "1"
result = find_by_id(table_name=table_name, id_field_name=id_field_name, id_value=id_value)
json_str = json.dumps(result, cls=ComplexEncoder)
print(json_str)

num = 1
update_field = "face"+ str(num)
update_sql = "update sys_user_verify set {face_field} = ? where user_id = ?".format(face_field=update_field)
update_sql = update_sql.replace(update_field + " = ?", update_field + " = %s").replace("user_id = ?", "user_id = %s")
print(update_sql)