/*
 * Copyright (c) 1996, 2020, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */
package java.sql;

import org.checkerframework.checker.nullness.qual.Nullable;
import java.math.BigDecimal;
import java.util.Calendar;
import java.io.Reader;
import java.io.InputStream;

public interface CallableStatement extends PreparedStatement {

    void registerOutParameter(int parameterIndex, int sqlType) throws SQLException;

    void registerOutParameter(int parameterIndex, int sqlType, int scale) throws SQLException;

    boolean wasNull() throws SQLException;

    @Nullable
    String getString(int parameterIndex) throws SQLException;

    boolean getBoolean(int parameterIndex) throws SQLException;

    byte getByte(int parameterIndex) throws SQLException;

    short getShort(int parameterIndex) throws SQLException;

    int getInt(int parameterIndex) throws SQLException;

    long getLong(int parameterIndex) throws SQLException;

    float getFloat(int parameterIndex) throws SQLException;

    double getDouble(int parameterIndex) throws SQLException;

    @Deprecated()
    @Nullable
    BigDecimal getBigDecimal(int parameterIndex, int scale) throws SQLException;

    byte @Nullable [] getBytes(int parameterIndex) throws SQLException;

    java.sql.@Nullable Date getDate(int parameterIndex) throws SQLException;

    java.sql.@Nullable Time getTime(int parameterIndex) throws SQLException;

    java.sql.@Nullable Timestamp getTimestamp(int parameterIndex) throws SQLException;

    @Nullable
    Object getObject(int parameterIndex) throws SQLException;

    @Nullable
    BigDecimal getBigDecimal(int parameterIndex) throws SQLException;

    @Nullable
    Object getObject(int parameterIndex, java.util.Map<String, Class<?>> map) throws SQLException;

    @Nullable
    Ref getRef(int parameterIndex) throws SQLException;

    @Nullable
    Blob getBlob(int parameterIndex) throws SQLException;

    @Nullable
    Clob getClob(int parameterIndex) throws SQLException;

    @Nullable
    Array getArray(int parameterIndex) throws SQLException;

    java.sql.@Nullable Date getDate(int parameterIndex, @Nullable Calendar cal) throws SQLException;

    java.sql.@Nullable Time getTime(int parameterIndex, @Nullable Calendar cal) throws SQLException;

    java.sql.@Nullable Timestamp getTimestamp(int parameterIndex, @Nullable Calendar cal) throws SQLException;

    void registerOutParameter(int parameterIndex, int sqlType, String typeName) throws SQLException;

    void registerOutParameter(String parameterName, int sqlType) throws SQLException;

    void registerOutParameter(String parameterName, int sqlType, int scale) throws SQLException;

    void registerOutParameter(String parameterName, int sqlType, String typeName) throws SQLException;

    java.net.@Nullable URL getURL(int parameterIndex) throws SQLException;

    void setURL(String parameterName, java.net.@Nullable URL val) throws SQLException;

    void setNull(String parameterName, int sqlType) throws SQLException;

    void setBoolean(String parameterName, boolean x) throws SQLException;

    void setByte(String parameterName, byte x) throws SQLException;

    void setShort(String parameterName, short x) throws SQLException;

    void setInt(String parameterName, int x) throws SQLException;

    void setLong(String parameterName, long x) throws SQLException;

    void setFloat(String parameterName, float x) throws SQLException;

    void setDouble(String parameterName, double x) throws SQLException;

    void setBigDecimal(String parameterName, @Nullable BigDecimal x) throws SQLException;

    void setString(String parameterName, @Nullable String x) throws SQLException;

    void setBytes(String parameterName, byte @Nullable [] x) throws SQLException;

    void setDate(String parameterName, java.sql.@Nullable Date x) throws SQLException;

    void setTime(String parameterName, java.sql.@Nullable Time x) throws SQLException;

    void setTimestamp(String parameterName, java.sql.@Nullable Timestamp x) throws SQLException;

    void setAsciiStream(String parameterName, java.io.@Nullable InputStream x, int length) throws SQLException;

    void setBinaryStream(String parameterName, java.io.@Nullable InputStream x, int length) throws SQLException;

    void setObject(String parameterName, @Nullable Object x, int targetSqlType, int scale) throws SQLException;

    void setObject(String parameterName, @Nullable Object x, int targetSqlType) throws SQLException;

    void setObject(String parameterName, @Nullable Object x) throws SQLException;

    void setCharacterStream(String parameterName, java.io.@Nullable Reader reader, int length) throws SQLException;

    void setDate(String parameterName, java.sql.@Nullable Date x, @Nullable Calendar cal) throws SQLException;

    void setTime(String parameterName, java.sql.@Nullable Time x, @Nullable Calendar cal) throws SQLException;

    void setTimestamp(String parameterName, java.sql.@Nullable Timestamp x, @Nullable Calendar cal) throws SQLException;

    void setNull(String parameterName, int sqlType, String typeName) throws SQLException;

    @Nullable
    String getString(String parameterName) throws SQLException;

    boolean getBoolean(String parameterName) throws SQLException;

    byte getByte(String parameterName) throws SQLException;

    short getShort(String parameterName) throws SQLException;

    int getInt(String parameterName) throws SQLException;

    long getLong(String parameterName) throws SQLException;

    float getFloat(String parameterName) throws SQLException;

    double getDouble(String parameterName) throws SQLException;

    byte @Nullable [] getBytes(String parameterName) throws SQLException;

    java.sql.@Nullable Date getDate(String parameterName) throws SQLException;

    java.sql.@Nullable Time getTime(String parameterName) throws SQLException;

    java.sql.@Nullable Timestamp getTimestamp(String parameterName) throws SQLException;

    @Nullable
    Object getObject(String parameterName) throws SQLException;

    @Nullable
    BigDecimal getBigDecimal(String parameterName) throws SQLException;

    @Nullable
    Object getObject(String parameterName, java.util.Map<String, Class<?>> map) throws SQLException;

    @Nullable
    Ref getRef(String parameterName) throws SQLException;

    @Nullable
    Blob getBlob(String parameterName) throws SQLException;

    @Nullable
    Clob getClob(String parameterName) throws SQLException;

    @Nullable
    Array getArray(String parameterName) throws SQLException;

    java.sql.@Nullable Date getDate(String parameterName, @Nullable Calendar cal) throws SQLException;

    java.sql.@Nullable Time getTime(String parameterName, @Nullable Calendar cal) throws SQLException;

    java.sql.@Nullable Timestamp getTimestamp(String parameterName, @Nullable Calendar cal) throws SQLException;

    java.net.@Nullable URL getURL(String parameterName) throws SQLException;

    @Nullable
    RowId getRowId(int parameterIndex) throws SQLException;

    @Nullable
    RowId getRowId(String parameterName) throws SQLException;

    void setRowId(String parameterName, @Nullable RowId x) throws SQLException;

    void setNString(String parameterName, @Nullable String value) throws SQLException;

    void setNCharacterStream(String parameterName, @Nullable Reader value, long length) throws SQLException;

    void setNClob(String parameterName, @Nullable NClob value) throws SQLException;

    void setClob(String parameterName, @Nullable Reader reader, long length) throws SQLException;

    void setBlob(String parameterName, @Nullable InputStream inputStream, long length) throws SQLException;

    void setNClob(String parameterName, @Nullable Reader reader, long length) throws SQLException;

    @Nullable
    NClob getNClob(int parameterIndex) throws SQLException;

    @Nullable
    NClob getNClob(String parameterName) throws SQLException;

    void setSQLXML(String parameterName, @Nullable SQLXML xmlObject) throws SQLException;

    @Nullable
    SQLXML getSQLXML(int parameterIndex) throws SQLException;

    @Nullable
    SQLXML getSQLXML(String parameterName) throws SQLException;

    @Nullable
    String getNString(int parameterIndex) throws SQLException;

    @Nullable
    String getNString(String parameterName) throws SQLException;

    java.io.@Nullable Reader getNCharacterStream(int parameterIndex) throws SQLException;

    java.io.@Nullable Reader getNCharacterStream(String parameterName) throws SQLException;

    java.io.@Nullable Reader getCharacterStream(int parameterIndex) throws SQLException;

    java.io.@Nullable Reader getCharacterStream(String parameterName) throws SQLException;

    void setBlob(String parameterName, @Nullable Blob x) throws SQLException;

    void setClob(String parameterName, @Nullable Clob x) throws SQLException;

    void setAsciiStream(String parameterName, java.io.@Nullable InputStream x, long length) throws SQLException;

    void setBinaryStream(String parameterName, java.io.@Nullable InputStream x, long length) throws SQLException;

    void setCharacterStream(String parameterName, java.io.@Nullable Reader reader, long length) throws SQLException;

    void setAsciiStream(String parameterName, java.io.@Nullable InputStream x) throws SQLException;

    void setBinaryStream(String parameterName, java.io.@Nullable InputStream x) throws SQLException;

    void setCharacterStream(String parameterName, java.io.@Nullable Reader reader) throws SQLException;

    void setNCharacterStream(String parameterName, @Nullable Reader value) throws SQLException;

    void setClob(String parameterName, @Nullable Reader reader) throws SQLException;

    void setBlob(String parameterName, @Nullable InputStream inputStream) throws SQLException;

    void setNClob(String parameterName, @Nullable Reader reader) throws SQLException;

    @Nullable
    public <T> T getObject(int parameterIndex, Class<T> type) throws SQLException;

    @Nullable
    public <T> T getObject(String parameterName, Class<T> type) throws SQLException;

    default void setObject(String parameterName, @Nullable Object x, SQLType targetSqlType, int scaleOrLength) throws SQLException;

    default void setObject(String parameterName, @Nullable Object x, SQLType targetSqlType) throws SQLException;

    default void registerOutParameter(int parameterIndex, SQLType sqlType) throws SQLException;

    default void registerOutParameter(int parameterIndex, SQLType sqlType, int scale) throws SQLException;

    default void registerOutParameter(int parameterIndex, SQLType sqlType, String typeName) throws SQLException;

    default void registerOutParameter(String parameterName, SQLType sqlType) throws SQLException;

    default void registerOutParameter(String parameterName, SQLType sqlType, int scale) throws SQLException;

    default void registerOutParameter(String parameterName, SQLType sqlType, String typeName) throws SQLException;
}
