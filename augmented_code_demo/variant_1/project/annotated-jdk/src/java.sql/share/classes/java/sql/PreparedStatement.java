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
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.math.BigDecimal;
    @Positive
import java.util.Calendar;
    @Positive
import java.io.Reader;
    @Positive
import java.io.InputStream;

    @Positive
@AnnotatedFor("nullness")
    @Positive
public interface PreparedStatement extends Statement {

    @Positive
    ResultSet executeQuery() throws SQLException;

    @Positive
    int executeUpdate() throws SQLException;

    @Positive
    void setNull(int parameterIndex, int sqlType) throws SQLException;

    @Positive
    void setBoolean(int parameterIndex, boolean x) throws SQLException;

    @Positive
    void setByte(int parameterIndex, byte x) throws SQLException;

    @Positive
    void setShort(int parameterIndex, short x) throws SQLException;

    @Positive
    void setInt(int parameterIndex, int x) throws SQLException;

    @Positive
    void setLong(int parameterIndex, long x) throws SQLException;

    @Positive
    void setFloat(int parameterIndex, float x) throws SQLException;

    @Positive
    void setDouble(int parameterIndex, double x) throws SQLException;

    @Positive
    void setBigDecimal(int parameterIndex, @Nullable BigDecimal x) throws SQLException;

    @Positive
    void setString(int parameterIndex, @Nullable String x) throws SQLException;

    @Positive
    void setBytes(int parameterIndex, byte @Nullable [] x) throws SQLException;

    @Positive
    void setDate(int parameterIndex, java.sql.@Nullable Date x) throws SQLException;

    @Positive
    void setTime(int parameterIndex, java.sql.@Nullable Time x) throws SQLException;

    @Positive
    void setTimestamp(int parameterIndex, java.sql.@Nullable Timestamp x) throws SQLException;

    @Positive
    void setAsciiStream(int parameterIndex, java.io.@Nullable InputStream x, int length) throws SQLException;

    @Positive
    @Deprecated()
    @Positive
    void setUnicodeStream(int parameterIndex, java.io.@Nullable InputStream x, int length) throws SQLException;

    @Positive
    void setBinaryStream(int parameterIndex, java.io.@Nullable InputStream x, int length) throws SQLException;

    @Positive
    void clearParameters() throws SQLException;

    @Positive
    void setObject(int parameterIndex, @Nullable Object x, int targetSqlType) throws SQLException;

    @Positive
    void setObject(int parameterIndex, @Nullable Object x) throws SQLException;

    @Positive
    boolean execute() throws SQLException;

    @Positive
    void addBatch() throws SQLException;

    @Positive
    void setCharacterStream(int parameterIndex, java.io.@Nullable Reader reader, int length) throws SQLException;

    @Positive
    void setRef(int parameterIndex, Ref x) throws SQLException;

    @Positive
    void setBlob(int parameterIndex, @Nullable Blob x) throws SQLException;

    @Positive
    void setClob(int parameterIndex, @Nullable Clob x) throws SQLException;

    @Positive
    void setArray(int parameterIndex, Array x) throws SQLException;

    @Positive
    @Nullable
    @Positive
    ResultSetMetaData getMetaData() throws SQLException;

    @Positive
    void setDate(int parameterIndex, java.sql.@Nullable Date x, @Nullable Calendar cal) throws SQLException;

    @Positive
    void setTime(int parameterIndex, java.sql.@Nullable Time x, @Nullable Calendar cal) throws SQLException;

    @Positive
    void setTimestamp(int parameterIndex, java.sql.@Nullable Timestamp x, @Nullable Calendar cal) throws SQLException;

    @Positive
    void setNull(int parameterIndex, int sqlType, String typeName) throws SQLException;

    @Positive
    void setURL(int parameterIndex, java.net.@Nullable URL x) throws SQLException;

    @Positive
    ParameterMetaData getParameterMetaData() throws SQLException;

    @Positive
    void setRowId(int parameterIndex, RowId x) throws SQLException;

    @Positive
    void setNString(int parameterIndex, @Nullable String value) throws SQLException;

    @Positive
    void setNCharacterStream(int parameterIndex, @Nullable Reader value, long length) throws SQLException;

    @Positive
    void setNClob(int parameterIndex, @Nullable NClob value) throws SQLException;

    @Positive
    void setClob(int parameterIndex, @Nullable Reader reader, long length) throws SQLException;

    @Positive
    void setBlob(int parameterIndex, @Nullable InputStream inputStream, long length) throws SQLException;

    @Positive
    void setNClob(int parameterIndex, @Nullable Reader reader, long length) throws SQLException;

    @Positive
    void setSQLXML(int parameterIndex, SQLXML xmlObject) throws SQLException;

    @Positive
    void setObject(int parameterIndex, @Nullable Object x, int targetSqlType, int scaleOrLength) throws SQLException;

    @Positive
    void setAsciiStream(int parameterIndex, java.io.@Nullable InputStream x, long length) throws SQLException;

    @Positive
    void setBinaryStream(int parameterIndex, java.io.@Nullable InputStream x, long length) throws SQLException;

    @Positive
    void setCharacterStream(int parameterIndex, java.io.@Nullable Reader reader, long length) throws SQLException;

    @Positive
    void setAsciiStream(int parameterIndex, java.io.@Nullable InputStream x) throws SQLException;

    @Positive
    void setBinaryStream(int parameterIndex, java.io.@Nullable InputStream x) throws SQLException;

    @Positive
    void setCharacterStream(int parameterIndex, java.io.@Nullable Reader reader) throws SQLException;

    @Positive
    void setNCharacterStream(int parameterIndex, @Nullable Reader value) throws SQLException;

    @Positive
    void setClob(int parameterIndex, @Nullable Reader reader) throws SQLException;

    @Positive
    void setBlob(int parameterIndex, @Nullable InputStream inputStream) throws SQLException;

    @Positive
    void setNClob(int parameterIndex, @Nullable Reader reader) throws SQLException;

    @Positive
    default void setObject(int parameterIndex, @Nullable Object x, SQLType targetSqlType, int scaleOrLength) throws SQLException;

    @Positive
    default void setObject(int parameterIndex, @Nullable Object x, SQLType targetSqlType) throws SQLException;

    @Positive
    default long executeLargeUpdate() throws SQLException;
    @Positive
}

// CFWR semantic augmentation - variant 1
