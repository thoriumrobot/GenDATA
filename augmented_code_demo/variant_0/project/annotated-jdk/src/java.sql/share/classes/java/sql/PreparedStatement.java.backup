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
import org.checkerframework.framework.qual.AnnotatedFor;
import java.math.BigDecimal;
import java.util.Calendar;
import java.io.Reader;
import java.io.InputStream;

@AnnotatedFor("nullness")
public interface PreparedStatement extends Statement {

    ResultSet executeQuery() throws SQLException;

    int executeUpdate() throws SQLException;

    void setNull(int parameterIndex, int sqlType) throws SQLException;

    void setBoolean(int parameterIndex, boolean x) throws SQLException;

    void setByte(int parameterIndex, byte x) throws SQLException;

    void setShort(int parameterIndex, short x) throws SQLException;

    void setInt(int parameterIndex, int x) throws SQLException;

    void setLong(int parameterIndex, long x) throws SQLException;

    void setFloat(int parameterIndex, float x) throws SQLException;

    void setDouble(int parameterIndex, double x) throws SQLException;

    void setBigDecimal(int parameterIndex, @Nullable BigDecimal x) throws SQLException;

    void setString(int parameterIndex, @Nullable String x) throws SQLException;

    void setBytes(int parameterIndex, byte @Nullable [] x) throws SQLException;

    void setDate(int parameterIndex, java.sql.@Nullable Date x) throws SQLException;

    void setTime(int parameterIndex, java.sql.@Nullable Time x) throws SQLException;

    void setTimestamp(int parameterIndex, java.sql.@Nullable Timestamp x) throws SQLException;

    void setAsciiStream(int parameterIndex, java.io.@Nullable InputStream x, int length) throws SQLException;

    @Deprecated()
    void setUnicodeStream(int parameterIndex, java.io.@Nullable InputStream x, int length) throws SQLException;

    void setBinaryStream(int parameterIndex, java.io.@Nullable InputStream x, int length) throws SQLException;

    void clearParameters() throws SQLException;

    void setObject(int parameterIndex, @Nullable Object x, int targetSqlType) throws SQLException;

    void setObject(int parameterIndex, @Nullable Object x) throws SQLException;

    boolean execute() throws SQLException;

    void addBatch() throws SQLException;

    void setCharacterStream(int parameterIndex, java.io.@Nullable Reader reader, int length) throws SQLException;

    void setRef(int parameterIndex, Ref x) throws SQLException;

    void setBlob(int parameterIndex, @Nullable Blob x) throws SQLException;

    void setClob(int parameterIndex, @Nullable Clob x) throws SQLException;

    void setArray(int parameterIndex, Array x) throws SQLException;

    @Nullable
    ResultSetMetaData getMetaData() throws SQLException;

    void setDate(int parameterIndex, java.sql.@Nullable Date x, @Nullable Calendar cal) throws SQLException;

    void setTime(int parameterIndex, java.sql.@Nullable Time x, @Nullable Calendar cal) throws SQLException;

    void setTimestamp(int parameterIndex, java.sql.@Nullable Timestamp x, @Nullable Calendar cal) throws SQLException;

    void setNull(int parameterIndex, int sqlType, String typeName) throws SQLException;

    void setURL(int parameterIndex, java.net.@Nullable URL x) throws SQLException;

    ParameterMetaData getParameterMetaData() throws SQLException;

    void setRowId(int parameterIndex, RowId x) throws SQLException;

    void setNString(int parameterIndex, @Nullable String value) throws SQLException;

    void setNCharacterStream(int parameterIndex, @Nullable Reader value, long length) throws SQLException;

    void setNClob(int parameterIndex, @Nullable NClob value) throws SQLException;

    void setClob(int parameterIndex, @Nullable Reader reader, long length) throws SQLException;

    void setBlob(int parameterIndex, @Nullable InputStream inputStream, long length) throws SQLException;

    void setNClob(int parameterIndex, @Nullable Reader reader, long length) throws SQLException;

    void setSQLXML(int parameterIndex, SQLXML xmlObject) throws SQLException;

    void setObject(int parameterIndex, @Nullable Object x, int targetSqlType, int scaleOrLength) throws SQLException;

    void setAsciiStream(int parameterIndex, java.io.@Nullable InputStream x, long length) throws SQLException;

    void setBinaryStream(int parameterIndex, java.io.@Nullable InputStream x, long length) throws SQLException;

    void setCharacterStream(int parameterIndex, java.io.@Nullable Reader reader, long length) throws SQLException;

    void setAsciiStream(int parameterIndex, java.io.@Nullable InputStream x) throws SQLException;

    void setBinaryStream(int parameterIndex, java.io.@Nullable InputStream x) throws SQLException;

    void setCharacterStream(int parameterIndex, java.io.@Nullable Reader reader) throws SQLException;

    void setNCharacterStream(int parameterIndex, @Nullable Reader value) throws SQLException;

    void setClob(int parameterIndex, @Nullable Reader reader) throws SQLException;

    void setBlob(int parameterIndex, @Nullable InputStream inputStream) throws SQLException;

    void setNClob(int parameterIndex, @Nullable Reader reader) throws SQLException;

    default void setObject(int parameterIndex, @Nullable Object x, SQLType targetSqlType, int scaleOrLength) throws SQLException;

    default void setObject(int parameterIndex, @Nullable Object x, SQLType targetSqlType) throws SQLException;

    default long executeLargeUpdate() throws SQLException;
}
