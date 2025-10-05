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
public interface ResultSet extends Wrapper, AutoCloseable {

    @Positive
    boolean next() throws SQLException;

    @Positive
    void close() throws SQLException;

    @Positive
    boolean wasNull() throws SQLException;

    @Positive
    @Nullable
    @Positive
    String getString(int columnIndex) throws SQLException;

    @Positive
    boolean getBoolean(int columnIndex) throws SQLException;

    @Positive
    byte getByte(int columnIndex) throws SQLException;

    @Positive
    short getShort(int columnIndex) throws SQLException;

    @Positive
    int getInt(int columnIndex) throws SQLException;

    @Positive
    long getLong(int columnIndex) throws SQLException;

    @Positive
    float getFloat(int columnIndex) throws SQLException;

    @Positive
    double getDouble(int columnIndex) throws SQLException;

    @Positive
    @Deprecated()
    @Positive
    @Nullable
    @Positive
    BigDecimal getBigDecimal(int columnIndex, int scale) throws SQLException;

    @Positive
    byte @Nullable [] getBytes(int columnIndex) throws SQLException;

    @Positive
    java.sql.@Nullable Date getDate(int columnIndex) throws SQLException;

    @Positive
    java.sql.@Nullable Time getTime(int columnIndex) throws SQLException;

    @Positive
    java.sql.@Nullable Timestamp getTimestamp(int columnIndex) throws SQLException;

    @Positive
    java.io.@Nullable InputStream getAsciiStream(int columnIndex) throws SQLException;

    @Positive
    @Deprecated()
    @Positive
    java.io.@Nullable InputStream getUnicodeStream(int columnIndex) throws SQLException;

    @Positive
    java.io.@Nullable InputStream getBinaryStream(int columnIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    String getString(String columnLabel) throws SQLException;

    @Positive
    boolean getBoolean(String columnLabel) throws SQLException;

    @Positive
    byte getByte(String columnLabel) throws SQLException;

    @Positive
    short getShort(String columnLabel) throws SQLException;

    @Positive
    int getInt(String columnLabel) throws SQLException;

    @Positive
    long getLong(String columnLabel) throws SQLException;

    @Positive
    float getFloat(String columnLabel) throws SQLException;

    @Positive
    double getDouble(String columnLabel) throws SQLException;

    @Positive
    @Deprecated()
    @Positive
    BigDecimal getBigDecimal(String columnLabel, int scale) throws SQLException;

    @Positive
    byte @Nullable [] getBytes(String columnLabel) throws SQLException;

    @Positive
    java.sql.@Nullable Date getDate(String columnLabel) throws SQLException;

    @Positive
    java.sql.@Nullable Time getTime(String columnLabel) throws SQLException;

    @Positive
    java.sql.@Nullable Timestamp getTimestamp(String columnLabel) throws SQLException;

    @Positive
    java.io.@Nullable InputStream getAsciiStream(String columnLabel) throws SQLException;

    @Positive
    @Deprecated()
    @Positive
    java.io.@Nullable InputStream getUnicodeStream(String columnLabel) throws SQLException;

    @Positive
    java.io.InputStream getBinaryStream(String columnLabel) throws SQLException;

    @Positive
    @Nullable
    @Positive
    SQLWarning getWarnings() throws SQLException;

    @Positive
    void clearWarnings() throws SQLException;

    @Positive
    String getCursorName() throws SQLException;

    @Positive
    ResultSetMetaData getMetaData() throws SQLException;

    @Positive
    @Nullable
    @Positive
    Object getObject(int columnIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Object getObject(String columnLabel) throws SQLException;

    @Positive
    int findColumn(String columnLabel) throws SQLException;

    @Positive
    java.io.@Nullable Reader getCharacterStream(int columnIndex) throws SQLException;

    @Positive
    java.io.@Nullable Reader getCharacterStream(String columnLabel) throws SQLException;

    @Positive
    @Nullable
    @Positive
    BigDecimal getBigDecimal(int columnIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    BigDecimal getBigDecimal(String columnLabel) throws SQLException;

    @Positive
    boolean isBeforeFirst() throws SQLException;

    @Positive
    boolean isAfterLast() throws SQLException;

    @Positive
    boolean isFirst() throws SQLException;

    @Positive
    boolean isLast() throws SQLException;

    @Positive
    void beforeFirst() throws SQLException;

    @Positive
    void afterLast() throws SQLException;

    @Positive
    boolean first() throws SQLException;

    @Positive
    boolean last() throws SQLException;

    @Positive
    int getRow() throws SQLException;

    @Positive
    boolean absolute(int row) throws SQLException;

    @Positive
    boolean relative(int rows) throws SQLException;

    @Positive
    boolean previous() throws SQLException;

    @Positive
    int FETCH_FORWARD;

    @Positive
    int FETCH_REVERSE;

    @Positive
    int FETCH_UNKNOWN;

    @Positive
    void setFetchDirection(int direction) throws SQLException;

    @Positive
    int getFetchDirection() throws SQLException;

    @Positive
    void setFetchSize(int rows) throws SQLException;

    @Positive
    int getFetchSize() throws SQLException;

    @Positive
    int TYPE_FORWARD_ONLY;

    @Positive
    int TYPE_SCROLL_INSENSITIVE;

    @Positive
    int TYPE_SCROLL_SENSITIVE;

    @Positive
    int getType() throws SQLException;

    @Positive
    int CONCUR_READ_ONLY;

    @Positive
    int CONCUR_UPDATABLE;

    @Positive
    int getConcurrency() throws SQLException;

    @Positive
    boolean rowUpdated() throws SQLException;

    @Positive
    boolean rowInserted() throws SQLException;

    @Positive
    boolean rowDeleted() throws SQLException;

    @Positive
    void updateNull(int columnIndex) throws SQLException;

    @Positive
    void updateBoolean(int columnIndex, boolean x) throws SQLException;

    @Positive
    void updateByte(int columnIndex, byte x) throws SQLException;

    @Positive
    void updateShort(int columnIndex, short x) throws SQLException;

    @Positive
    void updateInt(int columnIndex, int x) throws SQLException;

    @Positive
    void updateLong(int columnIndex, long x) throws SQLException;

    @Positive
    void updateFloat(int columnIndex, float x) throws SQLException;

    @Positive
    void updateDouble(int columnIndex, double x) throws SQLException;

    @Positive
    void updateBigDecimal(int columnIndex, @Nullable BigDecimal x) throws SQLException;

    @Positive
    void updateString(int columnIndex, @Nullable String x) throws SQLException;

    @Positive
    void updateBytes(int columnIndex, byte @Nullable [] x) throws SQLException;

    @Positive
    void updateDate(int columnIndex, java.sql.@Nullable Date x) throws SQLException;

    @Positive
    void updateTime(int columnIndex, java.sql.@Nullable Time x) throws SQLException;

    @Positive
    void updateTimestamp(int columnIndex, java.sql.@Nullable Timestamp x) throws SQLException;

    @Positive
    void updateAsciiStream(int columnIndex, java.io.@Nullable InputStream x, int length) throws SQLException;

    @Positive
    void updateBinaryStream(int columnIndex, java.io.@Nullable InputStream x, int length) throws SQLException;

    @Positive
    void updateCharacterStream(int columnIndex, java.io.@Nullable Reader x, int length) throws SQLException;

    @Positive
    void updateObject(int columnIndex, @Nullable Object x, int scaleOrLength) throws SQLException;

    @Positive
    void updateObject(int columnIndex, @Nullable Object x) throws SQLException;

    @Positive
    void updateNull(String columnLabel) throws SQLException;

    @Positive
    void updateBoolean(String columnLabel, boolean x) throws SQLException;

    @Positive
    void updateByte(String columnLabel, byte x) throws SQLException;

    @Positive
    void updateShort(String columnLabel, short x) throws SQLException;

    @Positive
    void updateInt(String columnLabel, int x) throws SQLException;

    @Positive
    void updateLong(String columnLabel, long x) throws SQLException;

    @Positive
    void updateFloat(String columnLabel, float x) throws SQLException;

    @Positive
    void updateDouble(String columnLabel, double x) throws SQLException;

    @Positive
    void updateBigDecimal(String columnLabel, @Nullable BigDecimal x) throws SQLException;

    @Positive
    void updateString(String columnLabel, @Nullable String x) throws SQLException;

    @Positive
    void updateBytes(String columnLabel, byte @Nullable [] x) throws SQLException;

    @Positive
    void updateDate(String columnLabel, java.sql.@Nullable Date x) throws SQLException;

    @Positive
    void updateTime(String columnLabel, java.sql.@Nullable Time x) throws SQLException;

    @Positive
    void updateTimestamp(String columnLabel, java.sql.@Nullable Timestamp x) throws SQLException;

    @Positive
    void updateAsciiStream(String columnLabel, java.io.@Nullable InputStream x, int length) throws SQLException;

    @Positive
    void updateBinaryStream(String columnLabel, java.io.@Nullable InputStream x, int length) throws SQLException;

    @Positive
    void updateCharacterStream(String columnLabel, java.io.@Nullable Reader reader, int length) throws SQLException;

    @Positive
    void updateObject(String columnLabel, @Nullable Object x, int scaleOrLength) throws SQLException;

    @Positive
    void updateObject(String columnLabel, @Nullable Object x) throws SQLException;

    @Positive
    void insertRow() throws SQLException;

    @Positive
    void updateRow() throws SQLException;

    @Positive
    void deleteRow() throws SQLException;

    @Positive
    void refreshRow() throws SQLException;

    @Positive
    void cancelRowUpdates() throws SQLException;

    @Positive
    void moveToInsertRow() throws SQLException;

    @Positive
    void moveToCurrentRow() throws SQLException;

    @Positive
    @Nullable
    @Positive
    Statement getStatement() throws SQLException;

    @Positive
    @Nullable
    @Positive
    Object getObject(int columnIndex, java.util.Map<String, Class<?>> map) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Ref getRef(int columnIndex) throws SQLException;

    @Positive
    Blob getBlob(int columnIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Clob getClob(int columnIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Array getArray(int columnIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Object getObject(String columnLabel, java.util.Map<String, Class<?>> map) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Ref getRef(String columnLabel) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Blob getBlob(String columnLabel) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Clob getClob(String columnLabel) throws SQLException;

    @Positive
    @Nullable
    @Positive
    Array getArray(String columnLabel) throws SQLException;

    @Positive
    java.sql.@Nullable Date getDate(int columnIndex, @Nullable Calendar cal) throws SQLException;

    @Positive
    java.sql.@Nullable Date getDate(String columnLabel, @Nullable Calendar cal) throws SQLException;

    @Positive
    java.sql.@Nullable Time getTime(int columnIndex, @Nullable Calendar cal) throws SQLException;

    @Positive
    java.sql.@Nullable Time getTime(String columnLabel, @Nullable Calendar cal) throws SQLException;

    @Positive
    java.sql.@Nullable Timestamp getTimestamp(int columnIndex, @Nullable Calendar cal) throws SQLException;

    @Positive
    java.sql.@Nullable Timestamp getTimestamp(String columnLabel, @Nullable Calendar cal) throws SQLException;

    @Positive
    int HOLD_CURSORS_OVER_COMMIT;

    @Positive
    int CLOSE_CURSORS_AT_COMMIT;

    @Positive
    java.net.@Nullable URL getURL(int columnIndex) throws SQLException;

    @Positive
    java.net.@Nullable URL getURL(String columnLabel) throws SQLException;

    @Positive
    void updateRef(int columnIndex, java.sql.@Nullable Ref x) throws SQLException;

    @Positive
    void updateRef(String columnLabel, java.sql.@Nullable Ref x) throws SQLException;

    @Positive
    void updateBlob(int columnIndex, java.sql.@Nullable Blob x) throws SQLException;

    @Positive
    void updateBlob(String columnLabel, java.sql.@Nullable Blob x) throws SQLException;

    @Positive
    void updateClob(int columnIndex, java.sql.@Nullable Clob x) throws SQLException;

    @Positive
    void updateClob(String columnLabel, java.sql.@Nullable Clob x) throws SQLException;

    @Positive
    void updateArray(int columnIndex, java.sql.@Nullable Array x) throws SQLException;

    @Positive
    void updateArray(String columnLabel, java.sql.@Nullable Array x) throws SQLException;

    @Positive
    @Nullable
    @Positive
    RowId getRowId(int columnIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    RowId getRowId(String columnLabel) throws SQLException;

    @Positive
    void updateRowId(int columnIndex, @Nullable RowId x) throws SQLException;

    @Positive
    void updateRowId(String columnLabel, @Nullable RowId x) throws SQLException;

    @Positive
    int getHoldability() throws SQLException;

    @Positive
    boolean isClosed() throws SQLException;

    @Positive
    void updateNString(int columnIndex, @Nullable String nString) throws SQLException;

    @Positive
    void updateNString(String columnLabel, @Nullable String nString) throws SQLException;

    @Positive
    void updateNClob(int columnIndex, @Nullable NClob nClob) throws SQLException;

    @Positive
    void updateNClob(String columnLabel, @Nullable NClob nClob) throws SQLException;

    @Positive
    @Nullable
    @Positive
    NClob getNClob(int columnIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    NClob getNClob(String columnLabel) throws SQLException;

    @Positive
    @Nullable
    @Positive
    SQLXML getSQLXML(int columnIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    SQLXML getSQLXML(String columnLabel) throws SQLException;

    @Positive
    void updateSQLXML(int columnIndex, @Nullable SQLXML xmlObject) throws SQLException;

    @Positive
    void updateSQLXML(String columnLabel, @Nullable SQLXML xmlObject) throws SQLException;

    @Positive
    @Nullable
    @Positive
    String getNString(int columnIndex) throws SQLException;

    @Positive
    @Nullable
    @Positive
    String getNString(String columnLabel) throws SQLException;

    @Positive
    java.io.@Nullable Reader getNCharacterStream(int columnIndex) throws SQLException;

    @Positive
    java.io.@Nullable Reader getNCharacterStream(String columnLabel) throws SQLException;

    @Positive
    void updateNCharacterStream(int columnIndex, java.io.@Nullable Reader x, long length) throws SQLException;

    @Positive
    void updateNCharacterStream(String columnLabel, java.io.@Nullable Reader reader, long length) throws SQLException;

    @Positive
    void updateAsciiStream(int columnIndex, java.io.@Nullable InputStream x, long length) throws SQLException;

    @Positive
    void updateBinaryStream(int columnIndex, java.io.@Nullable InputStream x, long length) throws SQLException;

    @Positive
    void updateCharacterStream(int columnIndex, java.io.@Nullable Reader x, long length) throws SQLException;

    @Positive
    void updateAsciiStream(String columnLabel, java.io.@Nullable InputStream x, long length) throws SQLException;

    @Positive
    void updateBinaryStream(String columnLabel, java.io.@Nullable InputStream x, long length) throws SQLException;

    @Positive
    void updateCharacterStream(String columnLabel, java.io.@Nullable Reader reader, long length) throws SQLException;

    @Positive
    void updateBlob(int columnIndex, @Nullable InputStream inputStream, long length) throws SQLException;

    @Positive
    void updateBlob(String columnLabel, @Nullable InputStream inputStream, long length) throws SQLException;

    @Positive
    void updateClob(int columnIndex, @Nullable Reader reader, long length) throws SQLException;

    @Positive
    void updateClob(String columnLabel, @Nullable Reader reader, long length) throws SQLException;

    @Positive
    void updateNClob(int columnIndex, @Nullable Reader reader, long length) throws SQLException;

    @Positive
    void updateNClob(String columnLabel, @Nullable Reader reader, long length) throws SQLException;

    @Positive
    void updateNCharacterStream(int columnIndex, java.io.@Nullable Reader x) throws SQLException;

    @Positive
    void updateNCharacterStream(String columnLabel, java.io.@Nullable Reader reader) throws SQLException;

    @Positive
    void updateAsciiStream(int columnIndex, java.io.InputStream x) throws SQLException;

    @Positive
    void updateBinaryStream(int columnIndex, java.io.@Nullable InputStream x) throws SQLException;

    @Positive
    void updateCharacterStream(int columnIndex, java.io.Reader x) throws SQLException;

    @Positive
    void updateAsciiStream(String columnLabel, java.io.@Nullable InputStream x) throws SQLException;

    @Positive
    void updateBinaryStream(String columnLabel, java.io.@Nullable InputStream x) throws SQLException;

    @Positive
    void updateCharacterStream(String columnLabel, java.io.@Nullable Reader reader) throws SQLException;

    @Positive
    void updateBlob(int columnIndex, @Nullable InputStream inputStream) throws SQLException;

    @Positive
    void updateBlob(String columnLabel, @Nullable InputStream inputStream) throws SQLException;

    @Positive
    void updateClob(int columnIndex, @Nullable Reader reader) throws SQLException;

    @Positive
    void updateClob(String columnLabel, @Nullable Reader reader) throws SQLException;

    @Positive
    void updateNClob(int columnIndex, @Nullable Reader reader) throws SQLException;

    @Positive
    void updateNClob(String columnLabel, @Nullable Reader reader) throws SQLException;

    @Positive
    @Nullable
    @Positive
    public <T> T getObject(int columnIndex, Class<T> type) throws SQLException;

    @Positive
    @Nullable
    @Positive
    public <T> T getObject(String columnLabel, Class<T> type) throws SQLException;

    @Positive
    default void updateObject(int columnIndex, @Nullable Object x, SQLType targetSqlType, int scaleOrLength) throws SQLException;

    @Positive
    default void updateObject(String columnLabel, @Nullable Object x, SQLType targetSqlType, int scaleOrLength) throws SQLException;

    @Positive
    default void updateObject(int columnIndex, @Nullable Object x, SQLType targetSqlType) throws SQLException;

    @Positive
    default void updateObject(String columnLabel, @Nullable Object x, SQLType targetSqlType) throws SQLException;
    @Positive
}

// CFWR semantic augmentation - variant 0
