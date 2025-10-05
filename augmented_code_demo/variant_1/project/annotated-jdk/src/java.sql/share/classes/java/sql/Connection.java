/*
    @Positive
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
    @Positive
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @Positive
 * This code is free software; you can redistribute it and/or modify it
    @Positive
 * under the terms of the GNU General Public License version 2 only, as
    @Positive
 * published by the Free Software Foundation.  Oracle designates this
    @Positive
 * particular file as subject to the "Classpath" exception as provided
    @Positive
 * by Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @Positive
 * This code is distributed in the hope that it will be useful, but WITHOUT
    @Positive
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @Positive
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @Positive
 * version 2 for more details (a copy is included in the LICENSE file that
    @Positive
 * accompanied this code).
    @Positive
 *
    @Positive
 * You should have received a copy of the GNU General Public License version
    @Positive
 * 2 along with this work; if not, write to the Free Software Foundation,
    @Positive
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Positive
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @Positive
 * or visit www.oracle.com if you need additional information or have any
    @Positive
 * questions.
    @Positive
 */
    @Positive
package java.sql;

    @Positive
import org.checkerframework.checker.mustcall.qual.InheritableMustCall;
    @Positive
import org.checkerframework.checker.sqlquotes.qual.SqlEvenQuotes;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Properties;
    @Positive
import java.util.concurrent.Executor;

    @Positive
@AnnotatedFor({ "mustcall" })
    @Positive
@InheritableMustCall("close")
    @Positive
public interface Connection extends Wrapper, AutoCloseable {

    @Positive
    Statement createStatement() throws SQLException;

    @Positive
    PreparedStatement prepareStatement(@SqlEvenQuotes String sql) throws SQLException;

    @Positive
    CallableStatement prepareCall(@SqlEvenQuotes String sql) throws SQLException;

    @Positive
    String nativeSQL(@SqlEvenQuotes String sql) throws SQLException;

    @Positive
    void setAutoCommit(boolean autoCommit) throws SQLException;

    @Positive
    boolean getAutoCommit() throws SQLException;

    @Positive
    void commit() throws SQLException;

    @Positive
    void rollback() throws SQLException;

    @Positive
    void close() throws SQLException;

    @Positive
    boolean isClosed() throws SQLException;

    @Positive
    DatabaseMetaData getMetaData() throws SQLException;

    @Positive
    void setReadOnly(boolean readOnly) throws SQLException;

    @Positive
    boolean isReadOnly() throws SQLException;

    @Positive
    void setCatalog(String catalog) throws SQLException;

    @Positive
    String getCatalog() throws SQLException;

    @Positive
    int TRANSACTION_NONE;

    @Positive
    int TRANSACTION_READ_UNCOMMITTED;

    @Positive
    int TRANSACTION_READ_COMMITTED;

    @Positive
    int TRANSACTION_REPEATABLE_READ;

    @Positive
    int TRANSACTION_SERIALIZABLE;

    @Positive
    void setTransactionIsolation(int level) throws SQLException;

    @Positive
    int getTransactionIsolation() throws SQLException;

    @Positive
    SQLWarning getWarnings() throws SQLException;

    @Positive
    void clearWarnings() throws SQLException;

    @Positive
    Statement createStatement(int resultSetType, int resultSetConcurrency) throws SQLException;

    @Positive
    PreparedStatement prepareStatement(@SqlEvenQuotes String sql, int resultSetType, int resultSetConcurrency) throws SQLException;

    @Positive
    CallableStatement prepareCall(@SqlEvenQuotes String sql, int resultSetType, int resultSetConcurrency) throws SQLException;

    @Positive
    java.util.Map<String, Class<?>> getTypeMap() throws SQLException;

    @Positive
    void setTypeMap(java.util.Map<String, Class<?>> map) throws SQLException;

    @Positive
    void setHoldability(int holdability) throws SQLException;

    @Positive
    int getHoldability() throws SQLException;

    @Positive
    Savepoint setSavepoint() throws SQLException;

    @Positive
    Savepoint setSavepoint(String name) throws SQLException;

    @Positive
    void rollback(Savepoint savepoint) throws SQLException;

    @Positive
    void releaseSavepoint(Savepoint savepoint) throws SQLException;

    @Positive
    Statement createStatement(int resultSetType, int resultSetConcurrency, int resultSetHoldability) throws SQLException;

    @Positive
    PreparedStatement prepareStatement(@SqlEvenQuotes String sql, int resultSetType, int resultSetConcurrency, int resultSetHoldability) throws SQLException;

    @Positive
    CallableStatement prepareCall(@SqlEvenQuotes String sql, int resultSetType, int resultSetConcurrency, int resultSetHoldability) throws SQLException;

    @Positive
    PreparedStatement prepareStatement(@SqlEvenQuotes String sql, int autoGeneratedKeys) throws SQLException;

    @Positive
    PreparedStatement prepareStatement(@SqlEvenQuotes String sql, int[] columnIndexes) throws SQLException;

    @Positive
    PreparedStatement prepareStatement(@SqlEvenQuotes String sql, String[] columnNames) throws SQLException;

    @Positive
    Clob createClob() throws SQLException;

    @Positive
    Blob createBlob() throws SQLException;

    @Positive
    NClob createNClob() throws SQLException;

    @Positive
    SQLXML createSQLXML() throws SQLException;

    @Positive
    boolean isValid(int timeout) throws SQLException;

    @Positive
    void setClientInfo(String name, String value) throws SQLClientInfoException;

    @Positive
    void setClientInfo(Properties properties) throws SQLClientInfoException;

    @Positive
    String getClientInfo(String name) throws SQLException;

    @Positive
    Properties getClientInfo() throws SQLException;

    @Positive
    Array createArrayOf(String typeName, Object[] elements) throws SQLException;

    @Positive
    Struct createStruct(String typeName, Object[] attributes) throws SQLException;

    @Positive
    void setSchema(String schema) throws SQLException;

    @Positive
    String getSchema() throws SQLException;

    @Positive
    void abort(Executor executor) throws SQLException;

    @Positive
    void setNetworkTimeout(Executor executor, int milliseconds) throws SQLException;

    @Positive
    int getNetworkTimeout() throws SQLException;

    @Positive
    default void beginRequest() throws SQLException;

    @Positive
    default void endRequest() throws SQLException;

    @Positive
    default boolean setShardingKeyIfValid(ShardingKey shardingKey, ShardingKey superShardingKey, int timeout) throws SQLException;

    @Positive
    default boolean setShardingKeyIfValid(ShardingKey shardingKey, int timeout) throws SQLException;

    @Positive
    default void setShardingKey(ShardingKey shardingKey, ShardingKey superShardingKey) throws SQLException;

    @Positive
    default void setShardingKey(ShardingKey shardingKey) throws SQLException;
    @Positive
}

// CFWR semantic augmentation - variant 1
