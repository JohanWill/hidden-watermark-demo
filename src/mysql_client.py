import pymysql
from fddeye.config import *
from fddeye.tools import logger
from DBUtils.PooledDB import PooledDB

class MysqlClient(object):
    pool = None;

    def __init__(self):
        self.db = DB
        self.host = DB_HOST
        self.port = DB_PORT
        self.user = DB_USER
        self.reset = DB_RESET
        self.passwd = DB_PASSWD
        self.blocking = DB_BLOCK
        self.maxusage = DB_MAX_USAGE
        self.mincached = DB_MIN_CACHED
        self.maxcached = DB_MAX_CACHED
        self.maxshared = DB_MAX_SHARED
        self.setsession = DB_SET_SESSION
        self.maxconnections = DB_MAX_CONN

        self.config = {
            'database':self.db,
            'host':self.host,
            'port':self.port,
            'user':self.user,
            'password':self.passwd,
            'charset':DB_CHARSET
        }

        if not self.pool:
            self.__class__.pool = PooledDB(creator=pymysql,mincached=self.mincached,maxcached=self.maxcached,
                        maxshared=self.maxshared,maxconnections=self.maxconnections,blocking=self.blocking,
                        maxusage=self.maxusage,setsession=self.setsession,reset=self.reset,**self.config)

        self.conn = None
        self.cursor = None
        self.__init_connection()
        logger.info(f"初始化{pymysql.__name__}连接池完毕.")

    def __init_connection(self):
        self.conn = self.pool.connection()
        self.cursor = self.conn.cursor()

    def close(self):
        try:
            self.cursor.close()
            self.conn.close()
        except Exception as e:
            print(e)

    def __execute(self, sql, param=()):
        count = self.cursor.execute(sql, param)
        return count

    def select_one(self, sql, param=()):
        """查询单个结果"""
        count = self.__execute(sql, param)
        result = self.cursor.fetchone()
        return count, result

    def select_many(self, sql, param=()):
        """
        查询多个结果
        :param sql: qsl语句
        :param param: sql参数
        :return: 结果数量和查询结果集
        """
        count = self.__execute(sql, param)
        result = self.cursor.fetchall()
        return count, result

    @property
    def autocommit(self):
        """开启自动提交"""
        self.execute("set autocommit = 1;")
    @property
    def commit(self):
        """提交"""
        self.conn.commit()
    @property
    def rollback(self):
        """回滚"""
        self.conn.rollback()

    def exexcute_update(self,sql):
        """执行更新语句"""
        try:
            count = self.__execute(sql)
            self.conn.commit()
        except Exception as e:
            logger.error(e)
            logger.error(f"Update Faild ! SQL is {sql}")
            self.conn.rollback()

    def columns(self,table):
        c,r = self.select_many(f"SELECT COLUMN_NAME FROM information_schema.`columns` WHERE TABLE_NAME = '{table}'")
        result = []
        for col in r:
            result.append(col[0])
        return result


    # def __del__(self):
    #     """重写类被清除时调用的方法
    #     """
    #     if self.cursor:
    #         self.cursor.close()
    #     if self.conn:
    #         self.conn.close()
    #     logger.info(u"数据库连接关闭")
if __name__ == '__main__':
    MysqlClient()