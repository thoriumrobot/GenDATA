/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.checker.signedness.qual.SignedPositive;
    @Positive
import org.checkerframework.checker.signedness.qual.Unsigned;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.nio.channels.FileChannel;
    @Positive
import jdk.internal.access.JavaIORandomAccessFileAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import sun.nio.ch.FileChannelImpl;

    @Positive
@AnnotatedFor({ "index", "interning", "mustcall", "nullness", "signedness" })
    @Positive
@UsesObjectEquals
    @Positive
public class RandomAccessFile implements DataOutput, DataInput, Closeable {

    @Positive
    public RandomAccessFile(String name, String mode) throws FileNotFoundException {
    @Positive
    }

    @Positive
    public RandomAccessFile(File file, String mode) throws FileNotFoundException {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public final FileDescriptor getFD(@MustCallAlias RandomAccessFile this) throws IOException;

    @Positive
    @MustCallAlias
    @Positive
    public final FileChannel getChannel(@MustCallAlias RandomAccessFile this);

    @Positive
    @GTENegativeOne
    @Positive
    public int read() throws IOException;

    @Positive
    @GTENegativeOne
    @Positive
    @LTEqLengthOf({ "#1" })
    @Positive
    public int read(@PolySigned byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
    @GTENegativeOne
    @Positive
    @LTEqLengthOf({ "#1" })
    @Positive
    public int read(@PolySigned byte[] b) throws IOException;

    @Positive
    public final void readFully(@PolySigned byte[] b) throws IOException;

    @Positive
    public final void readFully(@PolySigned byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
    @NonNegative
    @Positive
    public int skipBytes(@NonNegative int n) throws IOException;

    @Positive
    public void write(@PolySigned int b) throws IOException;

    @Positive
    public void write(@PolySigned byte[] b) throws IOException;

    @Positive
    public void write(@PolySigned byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
    public native long getFilePointer() throws IOException;

    @Positive
    public void seek(@NonNegative long pos) throws IOException;

    @Positive
    @NonNegative
    @Positive
    public native long length() throws IOException;

    @Positive
    public native void setLength(@NonNegative long newLength) throws IOException;

    @Positive
    public void close() throws IOException;

    @Positive
    public final boolean readBoolean() throws IOException;

    @Positive
    public final byte readByte() throws IOException;

    @Positive
    @NonNegative
    @Positive
    @SignedPositive
    @Positive
    public final int readUnsignedByte() throws IOException;

    @Positive
    public final short readShort() throws IOException;

    @Positive
    @NonNegative
    @Positive
    @SignedPositive
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
    @Nullable
    @Positive
    public final String readLine() throws IOException;

    @Positive
    public final String readUTF() throws IOException;

    @Positive
    public final void writeBoolean(boolean v) throws IOException;

    @Positive
    public final void writeByte(@PolySigned int v) throws IOException;

    @Positive
    public final void writeShort(@PolySigned int v) throws IOException;

    @Positive
    public final void writeChar(@PolySigned int v) throws IOException;

    @Positive
    public final void writeInt(@PolySigned int v) throws IOException;

    @Positive
    public final void writeLong(@PolySigned long v) throws IOException;

    @Positive
    public final void writeFloat(float v) throws IOException;

    @Positive
    public final void writeDouble(double v) throws IOException;

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public final void writeBytes(String s) throws IOException;

    @Positive
    public final void writeChars(String s) throws IOException;

    @Positive
    public final void writeUTF(String str) throws IOException;
    @Positive
}
