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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
public class Types {

    @Positive
    public static final int BIT;

    @Positive
    public static final int TINYINT;

    @Positive
    public static final int SMALLINT;

    @Positive
    public static final int INTEGER;

    @Positive
    public static final int BIGINT;

    @Positive
    public static final int FLOAT;

    @Positive
    public static final int REAL;

    @Positive
    public static final int DOUBLE;

    @Positive
    public static final int NUMERIC;

    @Positive
    public static final int DECIMAL;

    @Positive
    public static final int CHAR;

    @Positive
    public static final int VARCHAR;

    @Positive
    public static final int LONGVARCHAR;

    @Positive
    public static final int DATE;

    @Positive
    public static final int TIME;

    @Positive
    public static final int TIMESTAMP;

    @Positive
    public static final int BINARY;

    @Positive
    public static final int VARBINARY;

    @Positive
    public static final int LONGVARBINARY;

    @Positive
    public static final int NULL;

    @Positive
    public static final int OTHER;

    @Positive
    public static final int JAVA_OBJECT;

    @Positive
    public static final int DISTINCT;

    @Positive
    public static final int STRUCT;

    @Positive
    public static final int ARRAY;

    @Positive
    public static final int BLOB;

    @Positive
    public static final int CLOB;

    @Positive
    public static final int REF;

    @Positive
    public static final int DATALINK;

    @Positive
    public static final int BOOLEAN;

    @Positive
    public static final int ROWID;

    @Positive
    public static final int NCHAR;

    @Positive
    public static final int NVARCHAR;

    @Positive
    public static final int LONGNVARCHAR;

    @Positive
    public static final int NCLOB;

    @Positive
    public static final int SQLXML;

    @Positive
    public static final int REF_CURSOR;

    @Positive
    public static final int TIME_WITH_TIMEZONE;

    @Positive
    public static final int TIMESTAMP_WITH_TIMEZONE;
    @Positive
}

// CFWR semantic augmentation - variant 1
