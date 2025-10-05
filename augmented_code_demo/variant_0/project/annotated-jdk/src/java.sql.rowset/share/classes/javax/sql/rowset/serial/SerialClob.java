/*
    @Positive
 * Copyright (c) 2003, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.sql.rowset.serial;

    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresNonNullIf;
    @Positive
import org.checkerframework.checker.nullness.qual.NonNull;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import java.sql.*;
    @Positive
import java.io.*;
    @Positive
import java.util.Arrays;

    @Positive
public class SerialClob implements Clob, Serializable, Cloneable {

    @Positive
    public SerialClob(char[] ch) throws SerialException, SQLException {
    @Positive
    }

    @Positive
    public SerialClob(Clob clob) throws SerialException, SQLException {
    @Positive
    }

    @Positive
    public long length() throws SerialException;

    @Positive
    public java.io.Reader getCharacterStream() throws SerialException;

    @Positive
    public java.io.InputStream getAsciiStream() throws SerialException, SQLException;

    @Positive
    public String getSubString(long pos, int length) throws SerialException;

    @Positive
    public long position(String searchStr, long start) throws SerialException, SQLException;

    @Positive
    public long position(Clob searchStr, long start) throws SerialException, SQLException;

    @Positive
    public int setString(long pos, String str) throws SerialException;

    @Positive
    public int setString(long pos, String str, int offset, int length) throws SerialException;

    @Positive
    public java.io.OutputStream setAsciiStream(long pos) throws SerialException, SQLException;

    @Positive
    public java.io.Writer setCharacterStream(long pos) throws SerialException, SQLException;

    @Positive
    public void truncate(long length) throws SerialException;

    @Positive
    public Reader getCharacterStream(long pos, long length) throws SQLException;

    @Positive
    public void free() throws SQLException;

    @Positive
    @Pure
    @Positive
    @EnsuresNonNullIf(expression = "#1", result = true)
    @Positive
    public boolean equals(@Nullable Object obj);

    @Positive
    public int hashCode();

    @Positive
    public Object clone();
    @Positive
}

// CFWR semantic augmentation - variant 0
