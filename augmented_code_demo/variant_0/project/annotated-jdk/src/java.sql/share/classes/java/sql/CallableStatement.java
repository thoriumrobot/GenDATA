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
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import java.math.BigDecimal;
    @Positive
import java.util.Calendar;
    @Positive
import java.io.Reader;
    @Positive
import java.io.InputStream;

    @Positive
public interface CallableStatement extends PreparedStatement {

    @Positive
    void registerOutParameter(int parameterIndex, int sqlType) throws SQLException;

    @Positive
    void registerOutParameter(int parameterIndex, int sqlType, int scale) throws SQLException;

    @Positive
    boolean wasNull() throws SQLException;

    @Positive
    @Nullable
    @Positive
    String getString(int parameterIndex) throws SQLException;

    @Positive
    boolean getBoolean(int parameterIndex) throws SQLException;

    @Positive
    byte getByte(int parameterIndex) throws SQLException;

    @Positive
    short getShort(int parameterIndex) throws SQLException;

    @Positive
    int getInt(int parameterIndex) throws SQLException;

    @Positive
    long getLong(int parameterIndex) throws SQLException;

    @Positive
    float getFloat(int parameterIndex) throws SQLException;

    @Positive
    double getDouble(int parameterIndex) throws SQLException;

    @Positive
    @Deprecated()
    @Positive
    @Nullable
    @Positive
    BigDecimal getBigDecimal(int parameterIndex, int scale) throws SQLException;

    @Positive
    byte @Nullable [] getBytes(int parameterIndex) throws SQLException;

    @Positive
    java.sql.@Nullable Date getDate(int parameterIndex) throws SQLException;

    @Positive
    java.sql.@Nullable Time getTime(int parameterIndex) throws SQLException;

    @Positive
    java.sql.@Nullable Timestamp getTimestamp(int parameterIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Object getObject(int parameterIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    BigDecimal getBigDecimal(int parameterIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Object getObject(int parameterIndex, java.util.Map<String, Class<?>> map) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Ref getRef(int parameterIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Blob getBlob(int parameterIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Clob getClob(int parameterIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Array getArray(int parameterIndex) throws SQLException;

    @Positive
    java.sql.@Nullable Date getDate(int parameterIndex, @Nullable Calendar cal) throws SQLException;

    @Positive
    java.sql.@Nullable Time getTime(int parameterIndex, @Nullable Calendar cal) throws SQLException;

    @Positive
    java.sql.@Nullable Timestamp getTimestamp(int parameterIndex, @Nullable Calendar cal) throws SQLException;

    @Positive
    void registerOutParameter(int parameterIndex, int sqlType, String typeName) throws SQLException;

    @Positive
    void registerOutParameter(String parameterName, int sqlType) throws SQLException;

    @Positive
    void registerOutParameter(String parameterName, int sqlType, int scale) throws SQLException;

    @Positive
    void registerOutParameter(String parameterName, int sqlType, String typeName) throws SQLException;

    @Positive
    java.net.@Nullable URL getURL(int parameterIndex) throws SQLException;

    @Positive
    void setURL(String parameterName, java.net.@Nullable URL val) throws SQLException;

    @Positive
    void setNull(String parameterName, int sqlType) throws SQLException;

    @Positive
    void setBoolean(String parameterName, boolean x) throws SQLException;

    @Positive
    void setByte(String parameterName, byte x) throws SQLException;

    @Positive
    void setShort(String parameterName, short x) throws SQLException;

    @Positive
    void setInt(String parameterName, int x) throws SQLException;

    @Positive
    void setLong(String parameterName, long x) throws SQLException;

    @Positive
    void setFloat(String parameterName, float x) throws SQLException;

    @Positive
    void setDouble(String parameterName, double x) throws SQLException;

    @Positive
    void setBigDecimal(String parameterName, @Nullable BigDecimal x) throws SQLException;

    @Positive
    void setString(String parameterName, @Nullable String x) throws SQLException;

    @Positive
    void setBytes(String parameterName, byte @Nullable [] x) throws SQLException;

    @Positive
    void setDate(String parameterName, java.sql.@Nullable Date x) throws SQLException;

    @Positive
    void setTime(String parameterName, java.sql.@Nullable Time x) throws SQLException;

    @Positive
    void setTimestamp(String parameterName, java.sql.@Nullable Timestamp x) throws SQLException;

    @Positive
    void setAsciiStream(String parameterName, java.io.@Nullable InputStream x, int length) throws SQLException;

    @Positive
    void setBinaryStream(String parameterName, java.io.@Nullable InputStream x, int length) throws SQLException;

    @Positive
    void setObject(String parameterName, @Nullable Object x, int targetSqlType, int scale) throws SQLException;

    @Positive
    void setObject(String parameterName, @Nullable Object x, int targetSqlType) throws SQLException;

    @Positive
    void setObject(String parameterName, @Nullable Object x) throws SQLException;

    @Positive
    void setCharacterStream(String parameterName, java.io.@Nullable Reader reader, int length) throws SQLException;

    @Positive
    void setDate(String parameterName, java.sql.@Nullable Date x, @Nullable Calendar cal) throws SQLException;

    @Positive
    void setTime(String parameterName, java.sql.@Nullable Time x, @Nullable Calendar cal) throws SQLException;

    @Positive
    void setTimestamp(String parameterName, java.sql.@Nullable Timestamp x, @Nullable Calendar cal) throws SQLException;

    @Positive
    void setNull(String parameterName, int sqlType, String typeName) throws SQLException;

    @Positive
    @Nullable
    @Positive
    String getString(String parameterName) throws SQLException;

    @Positive
    boolean getBoolean(String parameterName) throws SQLException;

    @Positive
    byte getByte(String parameterName) throws SQLException;

    @Positive
    short getShort(String parameterName) throws SQLException;

    @Positive
    int getInt(String parameterName) throws SQLException;

    @Positive
    long getLong(String parameterName) throws SQLException;

    @Positive
    float getFloat(String parameterName) throws SQLException;

    @Positive
    double getDouble(String parameterName) throws SQLException;

    @Positive
    byte @Nullable [] getBytes(String parameterName) throws SQLException;

    @Positive
    java.sql.@Nullable Date getDate(String parameterName) throws SQLException;

    @Positive
    java.sql.@Nullable Time getTime(String parameterName) throws SQLException;

    @Positive
    java.sql.@Nullable Timestamp getTimestamp(String parameterName) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Object getObject(String parameterName) throws SQLException;

    @Positive
    @Nullable
    @Positive
    BigDecimal getBigDecimal(String parameterName) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Object getObject(String parameterName, java.util.Map<String, Class<?>> map) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Ref getRef(String parameterName) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Blob getBlob(String parameterName) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Clob getClob(String parameterName) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Array getArray(String parameterName) throws SQLException;

    @Positive
    java.sql.@Nullable Date getDate(String parameterName, @Nullable Calendar cal) throws SQLException;

    @Positive
    java.sql.@Nullable Time getTime(String parameterName, @Nullable Calendar cal) throws SQLException;

    @Positive
    java.sql.@Nullable Timestamp getTimestamp(String parameterName, @Nullable Calendar cal) throws SQLException;

    @Positive
    java.net.@Nullable URL getURL(String parameterName) throws SQLException;

    @Positive
    @Nullable
    @Positive
    RowId getRowId(int parameterIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    RowId getRowId(String parameterName) throws SQLException;

    @Positive
    void setRowId(String parameterName, @Nullable RowId x) throws SQLException;

    @Positive
    void setNString(String parameterName, @Nullable String value) throws SQLException;

    @Positive
    void setNCharacterStream(String parameterName, @Nullable Reader value, long length) throws SQLException;

    @Positive
    void setNClob(String parameterName, @Nullable NClob value) throws SQLException;

    @Positive
    void setClob(String parameterName, @Nullable Reader reader, long length) throws SQLException;

    @Positive
    void setBlob(String parameterName, @Nullable InputStream inputStream, long length) throws SQLException;

    @Positive
    void setNClob(String parameterName, @Nullable Reader reader, long length) throws SQLException;

    @Positive
    @Nullable
    @Positive
    NClob getNClob(int parameterIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    NClob getNClob(String parameterName) throws SQLException;

    @Positive
    void setSQLXML(String parameterName, @Nullable SQLXML xmlObject) throws SQLException;

    @Positive
    @Nullable
    @Positive
    SQLXML getSQLXML(int parameterIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    SQLXML getSQLXML(String parameterName) throws SQLException;

    @Positive
    @Nullable
    @Positive
    String getNString(int parameterIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    String getNString(String parameterName) throws SQLException;

    @Positive
    java.io.@Nullable Reader getNCharacterStream(int parameterIndex) throws SQLException;

    @Positive
    java.io.@Nullable Reader getNCharacterStream(String parameterName) throws SQLException;

    @Positive
    java.io.@Nullable Reader getCharacterStream(int parameterIndex) throws SQLException;

    @Positive
    java.io.@Nullable Reader getCharacterStream(String parameterName) throws SQLException;

    @Positive
    void setBlob(String parameterName, @Nullable Blob x) throws SQLException;

    @Positive
    void setClob(String parameterName, @Nullable Clob x) throws SQLException;

    @Positive
    void setAsciiStream(String parameterName, java.io.@Nullable InputStream x, long length) throws SQLException;

    @Positive
    void setBinaryStream(String parameterName, java.io.@Nullable InputStream x, long length) throws SQLException;

    @Positive
    void setCharacterStream(String parameterName, java.io.@Nullable Reader reader, long length) throws SQLException;

    @Positive
    void setAsciiStream(String parameterName, java.io.@Nullable InputStream x) throws SQLException;

    @Positive
    void setBinaryStream(String parameterName, java.io.@Nullable InputStream x) throws SQLException;

    @Positive
    void setCharacterStream(String parameterName, java.io.@Nullable Reader reader) throws SQLException;

    @Positive
    void setNCharacterStream(String parameterName, @Nullable Reader value) throws SQLException;

    @Positive
    void setClob(String parameterName, @Nullable Reader reader) throws SQLException;

    @Positive
    void setBlob(String parameterName, @Nullable InputStream inputStream) throws SQLException;

    @Positive
    void setNClob(String parameterName, @Nullable Reader reader) throws SQLException;

    @Positive
    @Nullable
    @Positive
    public <T> T getObject(int parameterIndex, Class<T> type) throws SQLException;

    @Positive
    @Nullable
    @Positive
    public <T> T getObject(String parameterName, Class<T> type) throws SQLException;

    @Positive
    default void setObject(String parameterName, @Nullable Object x, SQLType targetSqlType, int scaleOrLength) throws SQLException;

    @Positive
    default void setObject(String parameterName, @Nullable Object x, SQLType targetSqlType) throws SQLException;

    @Positive
    default void registerOutParameter(int parameterIndex, SQLType sqlType) throws SQLException;

    @Positive
    default void registerOutParameter(int parameterIndex, SQLType sqlType, int scale) throws SQLException;

    @Positive
    default void registerOutParameter(int parameterIndex, SQLType sqlType, String typeName) throws SQLException;

    @Positive
    default void registerOutParameter(String parameterName, SQLType sqlType) throws SQLException;

    @Positive
    default void registerOutParameter(String parameterName, SQLType sqlType, int scale) throws SQLException;

    @Positive
    default void registerOutParameter(String parameterName, SQLType sqlType, String typeName) throws SQLException;
    @Positive
}

// CFWR semantic augmentation - variant 0
