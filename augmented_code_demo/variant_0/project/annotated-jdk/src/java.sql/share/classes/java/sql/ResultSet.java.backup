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
public interface ResultSet extends Wrapper, AutoCloseable {

    boolean next() throws SQLException;

    void close() throws SQLException;

    boolean wasNull() throws SQLException;

    @Nullable
    String getString(int columnIndex) throws SQLException;

    boolean getBoolean(int columnIndex) throws SQLException;

    byte getByte(int columnIndex) throws SQLException;

    short getShort(int columnIndex) throws SQLException;

    int getInt(int columnIndex) throws SQLException;

    long getLong(int columnIndex) throws SQLException;

    float getFloat(int columnIndex) throws SQLException;

    double getDouble(int columnIndex) throws SQLException;

    @Deprecated()
    @Nullable
    BigDecimal getBigDecimal(int columnIndex, int scale) throws SQLException;

    byte @Nullable [] getBytes(int columnIndex) throws SQLException;

    java.sql.@Nullable Date getDate(int columnIndex) throws SQLException;

    java.sql.@Nullable Time getTime(int columnIndex) throws SQLException;

    java.sql.@Nullable Timestamp getTimestamp(int columnIndex) throws SQLException;

    java.io.@Nullable InputStream getAsciiStream(int columnIndex) throws SQLException;

    @Deprecated()
    java.io.@Nullable InputStream getUnicodeStream(int columnIndex) throws SQLException;

    java.io.@Nullable InputStream getBinaryStream(int columnIndex) throws SQLException;

    @Nullable
    String getString(String columnLabel) throws SQLException;

    boolean getBoolean(String columnLabel) throws SQLException;

    byte getByte(String columnLabel) throws SQLException;

    short getShort(String columnLabel) throws SQLException;

    int getInt(String columnLabel) throws SQLException;

    long getLong(String columnLabel) throws SQLException;

    float getFloat(String columnLabel) throws SQLException;

    double getDouble(String columnLabel) throws SQLException;

    @Deprecated()
    BigDecimal getBigDecimal(String columnLabel, int scale) throws SQLException;

    byte @Nullable [] getBytes(String columnLabel) throws SQLException;

    java.sql.@Nullable Date getDate(String columnLabel) throws SQLException;

    java.sql.@Nullable Time getTime(String columnLabel) throws SQLException;

    java.sql.@Nullable Timestamp getTimestamp(String columnLabel) throws SQLException;

    java.io.@Nullable InputStream getAsciiStream(String columnLabel) throws SQLException;

    @Deprecated()
    java.io.@Nullable InputStream getUnicodeStream(String columnLabel) throws SQLException;

    java.io.InputStream getBinaryStream(String columnLabel) throws SQLException;

    @Nullable
    SQLWarning getWarnings() throws SQLException;

    void clearWarnings() throws SQLException;

    String getCursorName() throws SQLException;

    ResultSetMetaData getMetaData() throws SQLException;

    @Nullable
    Object getObject(int columnIndex) throws SQLException;

    @Nullable
    Object getObject(String columnLabel) throws SQLException;

    int findColumn(String columnLabel) throws SQLException;

    java.io.@Nullable Reader getCharacterStream(int columnIndex) throws SQLException;

    java.io.@Nullable Reader getCharacterStream(String columnLabel) throws SQLException;

    @Nullable
    BigDecimal getBigDecimal(int columnIndex) throws SQLException;

    @Nullable
    BigDecimal getBigDecimal(String columnLabel) throws SQLException;

    boolean isBeforeFirst() throws SQLException;

    boolean isAfterLast() throws SQLException;

    boolean isFirst() throws SQLException;

    boolean isLast() throws SQLException;

    void beforeFirst() throws SQLException;

    void afterLast() throws SQLException;

    boolean first() throws SQLException;

    boolean last() throws SQLException;

    int getRow() throws SQLException;

    boolean absolute(int row) throws SQLException;

    boolean relative(int rows) throws SQLException;

    boolean previous() throws SQLException;

    int FETCH_FORWARD;

    int FETCH_REVERSE;

    int FETCH_UNKNOWN;

    void setFetchDirection(int direction) throws SQLException;

    int getFetchDirection() throws SQLException;

    void setFetchSize(int rows) throws SQLException;

    int getFetchSize() throws SQLException;

    int TYPE_FORWARD_ONLY;

    int TYPE_SCROLL_INSENSITIVE;

    int TYPE_SCROLL_SENSITIVE;

    int getType() throws SQLException;

    int CONCUR_READ_ONLY;

    int CONCUR_UPDATABLE;

    int getConcurrency() throws SQLException;

    boolean rowUpdated() throws SQLException;

    boolean rowInserted() throws SQLException;

    boolean rowDeleted() throws SQLException;

    void updateNull(int columnIndex) throws SQLException;

    void updateBoolean(int columnIndex, boolean x) throws SQLException;

    void updateByte(int columnIndex, byte x) throws SQLException;

    void updateShort(int columnIndex, short x) throws SQLException;

    void updateInt(int columnIndex, int x) throws SQLException;

    void updateLong(int columnIndex, long x) throws SQLException;

    void updateFloat(int columnIndex, float x) throws SQLException;

    void updateDouble(int columnIndex, double x) throws SQLException;

    void updateBigDecimal(int columnIndex, @Nullable BigDecimal x) throws SQLException;

    void updateString(int columnIndex, @Nullable String x) throws SQLException;

    void updateBytes(int columnIndex, byte @Nullable [] x) throws SQLException;

    void updateDate(int columnIndex, java.sql.@Nullable Date x) throws SQLException;

    void updateTime(int columnIndex, java.sql.@Nullable Time x) throws SQLException;

    void updateTimestamp(int columnIndex, java.sql.@Nullable Timestamp x) throws SQLException;

    void updateAsciiStream(int columnIndex, java.io.@Nullable InputStream x, int length) throws SQLException;

    void updateBinaryStream(int columnIndex, java.io.@Nullable InputStream x, int length) throws SQLException;

    void updateCharacterStream(int columnIndex, java.io.@Nullable Reader x, int length) throws SQLException;

    void updateObject(int columnIndex, @Nullable Object x, int scaleOrLength) throws SQLException;

    void updateObject(int columnIndex, @Nullable Object x) throws SQLException;

    void updateNull(String columnLabel) throws SQLException;

    void updateBoolean(String columnLabel, boolean x) throws SQLException;

    void updateByte(String columnLabel, byte x) throws SQLException;

    void updateShort(String columnLabel, short x) throws SQLException;

    void updateInt(String columnLabel, int x) throws SQLException;

    void updateLong(String columnLabel, long x) throws SQLException;

    void updateFloat(String columnLabel, float x) throws SQLException;

    void updateDouble(String columnLabel, double x) throws SQLException;

    void updateBigDecimal(String columnLabel, @Nullable BigDecimal x) throws SQLException;

    void updateString(String columnLabel, @Nullable String x) throws SQLException;

    void updateBytes(String columnLabel, byte @Nullable [] x) throws SQLException;

    void updateDate(String columnLabel, java.sql.@Nullable Date x) throws SQLException;

    void updateTime(String columnLabel, java.sql.@Nullable Time x) throws SQLException;

    void updateTimestamp(String columnLabel, java.sql.@Nullable Timestamp x) throws SQLException;

    void updateAsciiStream(String columnLabel, java.io.@Nullable InputStream x, int length) throws SQLException;

    void updateBinaryStream(String columnLabel, java.io.@Nullable InputStream x, int length) throws SQLException;

    void updateCharacterStream(String columnLabel, java.io.@Nullable Reader reader, int length) throws SQLException;

    void updateObject(String columnLabel, @Nullable Object x, int scaleOrLength) throws SQLException;

    void updateObject(String columnLabel, @Nullable Object x) throws SQLException;

    void insertRow() throws SQLException;

    void updateRow() throws SQLException;

    void deleteRow() throws SQLException;

    void refreshRow() throws SQLException;

    void cancelRowUpdates() throws SQLException;

    void moveToInsertRow() throws SQLException;

    void moveToCurrentRow() throws SQLException;

    @Nullable
    Statement getStatement() throws SQLException;

    @Nullable
    Object getObject(int columnIndex, java.util.Map<String, Class<?>> map) throws SQLException;

    @Nullable
    Ref getRef(int columnIndex) throws SQLException;

    Blob getBlob(int columnIndex) throws SQLException;

    @Nullable
    Clob getClob(int columnIndex) throws SQLException;

    @Nullable
    Array getArray(int columnIndex) throws SQLException;

    @Nullable
    Object getObject(String columnLabel, java.util.Map<String, Class<?>> map) throws SQLException;

    @Nullable
    Ref getRef(String columnLabel) throws SQLException;

    @Nullable
    Blob getBlob(String columnLabel) throws SQLException;

    @Nullable
    Clob getClob(String columnLabel) throws SQLException;

    @Nullable
    Array getArray(String columnLabel) throws SQLException;

    java.sql.@Nullable Date getDate(int columnIndex, @Nullable Calendar cal) throws SQLException;

    java.sql.@Nullable Date getDate(String columnLabel, @Nullable Calendar cal) throws SQLException;

    java.sql.@Nullable Time getTime(int columnIndex, @Nullable Calendar cal) throws SQLException;

    java.sql.@Nullable Time getTime(String columnLabel, @Nullable Calendar cal) throws SQLException;

    java.sql.@Nullable Timestamp getTimestamp(int columnIndex, @Nullable Calendar cal) throws SQLException;

    java.sql.@Nullable Timestamp getTimestamp(String columnLabel, @Nullable Calendar cal) throws SQLException;

    int HOLD_CURSORS_OVER_COMMIT;

    int CLOSE_CURSORS_AT_COMMIT;

    java.net.@Nullable URL getURL(int columnIndex) throws SQLException;

    java.net.@Nullable URL getURL(String columnLabel) throws SQLException;

    void updateRef(int columnIndex, java.sql.@Nullable Ref x) throws SQLException;

    void updateRef(String columnLabel, java.sql.@Nullable Ref x) throws SQLException;

    void updateBlob(int columnIndex, java.sql.@Nullable Blob x) throws SQLException;

    void updateBlob(String columnLabel, java.sql.@Nullable Blob x) throws SQLException;

    void updateClob(int columnIndex, java.sql.@Nullable Clob x) throws SQLException;

    void updateClob(String columnLabel, java.sql.@Nullable Clob x) throws SQLException;

    void updateArray(int columnIndex, java.sql.@Nullable Array x) throws SQLException;

    void updateArray(String columnLabel, java.sql.@Nullable Array x) throws SQLException;

    @Nullable
    RowId getRowId(int columnIndex) throws SQLException;

    @Nullable
    RowId getRowId(String columnLabel) throws SQLException;

    void updateRowId(int columnIndex, @Nullable RowId x) throws SQLException;

    void updateRowId(String columnLabel, @Nullable RowId x) throws SQLException;

    int getHoldability() throws SQLException;

    boolean isClosed() throws SQLException;

    void updateNString(int columnIndex, @Nullable String nString) throws SQLException;

    void updateNString(String columnLabel, @Nullable String nString) throws SQLException;

    void updateNClob(int columnIndex, @Nullable NClob nClob) throws SQLException;

    void updateNClob(String columnLabel, @Nullable NClob nClob) throws SQLException;

    @Nullable
    NClob getNClob(int columnIndex) throws SQLException;

    @Nullable
    NClob getNClob(String columnLabel) throws SQLException;

    @Nullable
    SQLXML getSQLXML(int columnIndex) throws SQLException;

    @Nullable
    SQLXML getSQLXML(String columnLabel) throws SQLException;

    void updateSQLXML(int columnIndex, @Nullable SQLXML xmlObject) throws SQLException;

    void updateSQLXML(String columnLabel, @Nullable SQLXML xmlObject) throws SQLException;

    @Nullable
    String getNString(int columnIndex) throws SQLException;

    @Nullable
    String getNString(String columnLabel) throws SQLException;

    java.io.@Nullable Reader getNCharacterStream(int columnIndex) throws SQLException;

    java.io.@Nullable Reader getNCharacterStream(String columnLabel) throws SQLException;

    void updateNCharacterStream(int columnIndex, java.io.@Nullable Reader x, long length) throws SQLException;

    void updateNCharacterStream(String columnLabel, java.io.@Nullable Reader reader, long length) throws SQLException;

    void updateAsciiStream(int columnIndex, java.io.@Nullable InputStream x, long length) throws SQLException;

    void updateBinaryStream(int columnIndex, java.io.@Nullable InputStream x, long length) throws SQLException;

    void updateCharacterStream(int columnIndex, java.io.@Nullable Reader x, long length) throws SQLException;

    void updateAsciiStream(String columnLabel, java.io.@Nullable InputStream x, long length) throws SQLException;

    void updateBinaryStream(String columnLabel, java.io.@Nullable InputStream x, long length) throws SQLException;

    void updateCharacterStream(String columnLabel, java.io.@Nullable Reader reader, long length) throws SQLException;

    void updateBlob(int columnIndex, @Nullable InputStream inputStream, long length) throws SQLException;

    void updateBlob(String columnLabel, @Nullable InputStream inputStream, long length) throws SQLException;

    void updateClob(int columnIndex, @Nullable Reader reader, long length) throws SQLException;

    void updateClob(String columnLabel, @Nullable Reader reader, long length) throws SQLException;

    void updateNClob(int columnIndex, @Nullable Reader reader, long length) throws SQLException;

    void updateNClob(String columnLabel, @Nullable Reader reader, long length) throws SQLException;

    void updateNCharacterStream(int columnIndex, java.io.@Nullable Reader x) throws SQLException;

    void updateNCharacterStream(String columnLabel, java.io.@Nullable Reader reader) throws SQLException;

    void updateAsciiStream(int columnIndex, java.io.InputStream x) throws SQLException;

    void updateBinaryStream(int columnIndex, java.io.@Nullable InputStream x) throws SQLException;

    void updateCharacterStream(int columnIndex, java.io.Reader x) throws SQLException;

    void updateAsciiStream(String columnLabel, java.io.@Nullable InputStream x) throws SQLException;

    void updateBinaryStream(String columnLabel, java.io.@Nullable InputStream x) throws SQLException;

    void updateCharacterStream(String columnLabel, java.io.@Nullable Reader reader) throws SQLException;

    void updateBlob(int columnIndex, @Nullable InputStream inputStream) throws SQLException;

    void updateBlob(String columnLabel, @Nullable InputStream inputStream) throws SQLException;

    void updateClob(int columnIndex, @Nullable Reader reader) throws SQLException;

    void updateClob(String columnLabel, @Nullable Reader reader) throws SQLException;

    void updateNClob(int columnIndex, @Nullable Reader reader) throws SQLException;

    void updateNClob(String columnLabel, @Nullable Reader reader) throws SQLException;

    @Nullable
    public <T> T getObject(int columnIndex, Class<T> type) throws SQLException;

    @Nullable
    public <T> T getObject(String columnLabel, Class<T> type) throws SQLException;

    default void updateObject(int columnIndex, @Nullable Object x, SQLType targetSqlType, int scaleOrLength) throws SQLException;

    default void updateObject(String columnLabel, @Nullable Object x, SQLType targetSqlType, int scaleOrLength) throws SQLException;

    default void updateObject(int columnIndex, @Nullable Object x, SQLType targetSqlType) throws SQLException;

    default void updateObject(String columnLabel, @Nullable Object x, SQLType targetSqlType) throws SQLException;
}
