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
import org.checkerframework.common.value.qual.IntVal;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.index.qual.Positive;

    @Positive
public interface ResultSetMetaData extends Wrapper {

    @Positive
    @NonNegative
    @Positive
    int getColumnCount() throws SQLException;

    @Positive
    boolean isAutoIncrement(@Positive int column) throws SQLException;

    @Positive
    boolean isCaseSensitive(@Positive int column) throws SQLException;

    @Positive
    boolean isSearchable(@Positive int column) throws SQLException;

    @Positive
    boolean isCurrency(@Positive int column) throws SQLException;

    @Positive
    @IntVal({ 0, 1, 2 })
    @Positive
    int isNullable(@Positive int column) throws SQLException;

    @Positive
    @IntVal(0)
    @Positive
    int columnNoNulls;

    @Positive
    @IntVal(1)
    @Positive
    int columnNullable;

    @Positive
    @IntVal(2)
    @Positive
    int columnNullableUnknown;

    @Positive
    boolean isSigned(@Positive int column) throws SQLException;

    @Positive
    @NonNegative
    @Positive
    int getColumnDisplaySize(@Positive int column) throws SQLException;

    @Positive
    String getColumnLabel(@Positive int column) throws SQLException;

    @Positive
    String getColumnName(@Positive int column) throws SQLException;

    @Positive
    String getSchemaName(@Positive int column) throws SQLException;

    @Positive
    @NonNegative
    @Positive
    int getPrecision(@Positive int column) throws SQLException;

    @Positive
    @NonNegative
    @Positive
    int getScale(@Positive int column) throws SQLException;

    @Positive
    String getTableName(@Positive int column) throws SQLException;

    @Positive
    String getCatalogName(@Positive int column) throws SQLException;

    @Positive
    int getColumnType(@Positive int column) throws SQLException;

    @Positive
    String getColumnTypeName(@Positive int column) throws SQLException;

    @Positive
    boolean isReadOnly(@Positive int column) throws SQLException;

    @Positive
    boolean isWritable(@Positive int column) throws SQLException;

    @Positive
    boolean isDefinitelyWritable(@Positive int column) throws SQLException;

    @Positive
    String getColumnClassName(@Positive int column) throws SQLException;
    @Positive
}

// CFWR semantic augmentation - variant 0
