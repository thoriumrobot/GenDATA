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

import org.checkerframework.common.value.qual.IntVal;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.index.qual.Positive;

public interface ResultSetMetaData extends Wrapper {

    @NonNegative
    int getColumnCount() throws SQLException;

    boolean isAutoIncrement(@Positive int column) throws SQLException;

    boolean isCaseSensitive(@Positive int column) throws SQLException;

    boolean isSearchable(@Positive int column) throws SQLException;

    boolean isCurrency(@Positive int column) throws SQLException;

    @IntVal({ 0, 1, 2 })
    int isNullable(@Positive int column) throws SQLException;

    @IntVal(0)
    int columnNoNulls;

    @IntVal(1)
    int columnNullable;

    @IntVal(2)
    int columnNullableUnknown;

    boolean isSigned(@Positive int column) throws SQLException;

    @NonNegative
    int getColumnDisplaySize(@Positive int column) throws SQLException;

    String getColumnLabel(@Positive int column) throws SQLException;

    String getColumnName(@Positive int column) throws SQLException;

    String getSchemaName(@Positive int column) throws SQLException;

    @NonNegative
    int getPrecision(@Positive int column) throws SQLException;

    @NonNegative
    int getScale(@Positive int column) throws SQLException;

    String getTableName(@Positive int column) throws SQLException;

    String getCatalogName(@Positive int column) throws SQLException;

    int getColumnType(@Positive int column) throws SQLException;

    String getColumnTypeName(@Positive int column) throws SQLException;

    boolean isReadOnly(@Positive int column) throws SQLException;

    boolean isWritable(@Positive int column) throws SQLException;

    boolean isDefinitelyWritable(@Positive int column) throws SQLException;

    String getColumnClassName(@Positive int column) throws SQLException;
}
