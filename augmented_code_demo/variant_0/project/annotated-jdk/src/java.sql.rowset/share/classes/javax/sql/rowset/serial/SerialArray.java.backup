/*
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.sql.rowset.serial;

import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
import org.checkerframework.checker.nullness.qual.NonNull;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import java.sql.*;
import java.io.*;
import java.util.Map;
import java.net.URL;
import java.util.Arrays;

public class SerialArray implements Array, Serializable, Cloneable {

    public SerialArray(Array array, Map<String, Class<?>> map) throws SerialException, SQLException {
    }

    public void free() throws SQLException;

    public SerialArray(Array array) throws SerialException, SQLException {
    }

    public Object getArray() throws SerialException;

    public Object getArray(Map<String, Class<?>> map) throws SerialException;

    public Object getArray(long index, int count) throws SerialException;

    public Object getArray(long index, int count, Map<String, Class<?>> map) throws SerialException;

    public int getBaseType() throws SerialException;

    public String getBaseTypeName() throws SerialException;

    public ResultSet getResultSet(long index, int count) throws SerialException;

    public ResultSet getResultSet(Map<String, Class<?>> map) throws SerialException;

    public ResultSet getResultSet() throws SerialException;

    public ResultSet getResultSet(long index, int count, Map<String, Class<?>> map) throws SerialException;

    @Pure
    @EnsuresNonNullIf(expression = "#1", result = true)
    public boolean equals(@Nullable Object obj);

    public int hashCode();

    public Object clone();
}
