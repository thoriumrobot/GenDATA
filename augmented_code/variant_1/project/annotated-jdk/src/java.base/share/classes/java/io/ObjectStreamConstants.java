/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
package java.io;

    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public interface ObjectStreamConstants {

    @Positive
    static final short STREAM_MAGIC;

    @Positive
    static final short STREAM_VERSION;

    @Positive
    static final byte TC_BASE;

    @Positive
    static final byte TC_NULL;

    @Positive
    static final byte TC_REFERENCE;

    @Positive
    static final byte TC_CLASSDESC;

    @Positive
    static final byte TC_OBJECT;

    @Positive
    static final byte TC_STRING;

    @Positive
    static final byte TC_ARRAY;

    @Positive
    static final byte TC_CLASS;

    @Positive
    static final byte TC_BLOCKDATA;

    @Positive
    static final byte TC_ENDBLOCKDATA;

    @Positive
    static final byte TC_RESET;

    @Positive
    static final byte TC_BLOCKDATALONG;

    @Positive
    static final byte TC_EXCEPTION;

    @Positive
    static final byte TC_LONGSTRING;

    @Positive
    static final byte TC_PROXYCLASSDESC;

    @Positive
    static final byte TC_ENUM;

    @Positive
    static final byte TC_MAX;

    @Positive
    static final int baseWireHandle;

    @Positive
    static final byte SC_WRITE_METHOD;

    @Positive
    static final byte SC_BLOCK_DATA;

    @Positive
    static final byte SC_SERIALIZABLE;

    @Positive
    static final byte SC_EXTERNALIZABLE;

    @Positive
    static final byte SC_ENUM;

    @Positive
    static final SerializablePermission SUBSTITUTION_PERMISSION;

    @Positive
    static final SerializablePermission SUBCLASS_IMPLEMENTATION_PERMISSION;

    @Positive
    static final SerializablePermission SERIAL_FILTER_PERMISSION;

    @Positive
    public static final int PROTOCOL_VERSION_1;

    @Positive
    public static final int PROTOCOL_VERSION_2;
    @Positive
}
