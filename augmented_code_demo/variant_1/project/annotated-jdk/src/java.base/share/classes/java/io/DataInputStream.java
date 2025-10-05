/*
    @Positive
 * Copyright (c) 1994, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.LTLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.checker.signedness.qual.SignedPositive;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.util.Objects;

    @Positive
@AnnotatedFor({ "index", "mustcall", "nullness", "signedness" })
    @Positive
public class DataInputStream extends FilterInputStream implements DataInput {

    @Positive
    @MustCallAlias
    @Positive
    public DataInputStream(@MustCallAlias InputStream in) {
    @Positive
    }

    @Positive
    @GTENegativeOne
    @Positive
    @LTEqLengthOf({ "#1" })
    @Positive
    public final int read(@PolySigned byte @Nullable [] b) throws IOException;

    @Positive
    @GTENegativeOne
    @Positive
    @LTEqLengthOf({ "#1" })
    @Positive
    public final int read(@PolySigned byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
    public final void readFully(@PolySigned byte[] b) throws IOException;

    @Positive
    public final void readFully(@PolySigned byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
    @NonNegative
    @Positive
    public final int skipBytes(int n) throws IOException;

    @Positive
    public final boolean readBoolean() throws IOException;

    @Positive
    public final byte readByte() throws IOException;

    @Positive
    @SignedPositive
    @Positive
    @NonNegative
    @Positive
    public final int readUnsignedByte() throws IOException;

    @Positive
    public final short readShort() throws IOException;

    @Positive
    @SignedPositive
    @Positive
    @NonNegative
    @Positive
    public final int readUnsignedShort() throws IOException;

    @Positive
    public final char readChar() throws IOException;

    @Positive
    public final int readInt() throws IOException;

    @Positive
    public final long readLong() throws IOException;

    @Positive
    public final float readFloat() throws IOException;

    @Positive
    public final double readDouble() throws IOException;

    @Positive
    @Deprecated
    @Positive
    @Nullable
    @Positive
    public final String readLine() throws IOException;

    @Positive
    public final String readUTF() throws IOException;

    @Positive
    public static final String readUTF(DataInput in) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 1
