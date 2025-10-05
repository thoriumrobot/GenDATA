/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
    @Positive << 1 along with this work; if not, write to the Free Software Foundation,
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
import org.checkerframework.checker.index.qual.IndexOrHigh;
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
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.ObjectStreamClass.WeakClassKey;
    @Positive
import java.lang.ref.ReferenceQueue;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.List;
    @Positive
import java.util.StringJoiner;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import static java.io.ObjectStreamClass.processQueue;
    @Positive
import sun.reflect.misc.ReflectUtil;

    @Positive
@AnnotatedFor({ "nullness", "index", "signedness" })
    @Positive
public class ObjectOutputStream extends OutputStream implements ObjectOutput, ObjectStreamConstants {

    @Positive
    private static class Caches {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public ObjectOutputStream(@MustCallAlias OutputStream out) throws IOException {
    @Positive
    }

    @Positive
    protected ObjectOutputStream() throws IOException, SecurityException {
    @Positive
    }

    @Positive
    public void useProtocolVersion(int version) throws IOException;

    @Positive
    public final void writeObject(@Nullable Object obj) throws IOException;

    @Positive
    protected void writeObjectOverride(Object obj) throws IOException;

    @Positive
    public void writeUnshared(@Nullable Object obj) throws IOException;

    @Positive
    public void defaultWriteObject() throws IOException;

    @Positive
    public ObjectOutputStream.PutField putFields() throws IOException;

    @Positive
    public void writeFields() throws IOException;

    @Positive
    public void reset() throws IOException;

    @Positive
    protected void annotateClass(Class<?> cl) throws IOException;

    @Positive
    protected void annotateProxyClass(Class<?> cl) throws IOException;

    @Positive
    protected Object replaceObject(Object obj) throws IOException;

    @Positive
    protected boolean enableReplaceObject(boolean enable) throws SecurityException;

    @Positive
    protected void writeStreamHeader() throws IOException;

    @Positive
    protected void writeClassDescriptor(ObjectStreamClass desc) throws IOException;

    @Positive
    public void write(@PolySigned int val) throws IOException;

    @Positive
    public void write(@PolySigned byte[] buf) throws IOException;

    @Positive
    public void write(@PolySigned byte[] buf, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
    public void flush() throws IOException;

    @Positive
    protected void drain() throws IOException;

    @Positive
    public void close() throws IOException;

    @Positive
    public void writeBoolean(boolean val) throws IOException;

    @Positive
    public void writeByte(int val) throws IOException;

    @Positive
    public void writeShort(int val) throws IOException;

    @Positive
    public void writeChar(int val) throws IOException;

    @Positive
    public void writeInt(int val) throws IOException;

    @Positive
    public void writeLong(long val) throws IOException;

    @Positive
    public void writeFloat(float val) throws IOException;

    @Positive
    public void writeDouble(double val) throws IOException;

    @Positive
    public void writeBytes(String str) throws IOException;

    @Positive
    public void writeChars(String str) throws IOException;

    @Positive
    public void writeUTF(String str) throws IOException;

    @Positive
    public abstract static class PutField {

    @Positive
        public PutField() {
    @Positive
        }

    @Positive
        public abstract void put(String name, boolean val);

    @Positive
        public abstract void put(String name, byte val);

    @Positive
        public abstract void put(String name, char val);

    @Positive
        public abstract void put(String name, short val);

    @Positive
        public abstract void put(String name, int val);

    @Positive
        public abstract void put(String name, long val);

    @Positive
        public abstract void put(String name, float val);

    @Positive
        public abstract void put(String name, double val);

    @Positive
        public abstract void put(String name, @Nullable Object val);

    @Positive
        @Deprecated
    @Positive
        public abstract void write(ObjectOutput out) throws IOException;
    @Positive
    }

    @Positive
    int getProtocolVersion();

    @Positive
    void writeTypeString(String str) throws IOException;

    @Positive
    private class PutFieldImpl extends PutField {

    @Positive
        public void put(String name, boolean val);

    @Positive
        public void put(String name, byte val);

    @Positive
        public void put(String name, char val);

    @Positive
        public void put(String name, short val);

    @Positive
        public void put(String name, int val);

    @Positive
        public void put(String name, float val);

    @Positive
        public void put(String name, long val);

    @Positive
        public void put(String name, double val);

    @Positive
        public void put(String name, Object val);

    @Positive
        public void write(ObjectOutput out) throws IOException;

    @Positive
        void writeFields() throws IOException;
    @Positive
    }

    @Positive
    private static class BlockDataOutputStream extends OutputStream implements DataOutput {

    @Positive
        boolean setBlockDataMode(boolean mode) throws IOException;

    @Positive
        boolean getBlockDataMode();

    @Positive
        public void write(int b) throws IOException;

    @Positive
        public void write(byte[] b) throws IOException;

    @Positive
        public void write(byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
        public void flush() throws IOException;

    @Positive
        public void close() throws IOException;

    @Positive
        void write(byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len, boolean copy) throws IOException;

    @Positive
        void drain() throws IOException;

    @Positive
        public void writeBoolean(boolean v) throws IOException;

    @Positive
        public void writeByte(int v) throws IOException;

    @Positive
        public void writeChar(int v) throws IOException;

    @Positive
        public void writeShort(int v) throws IOException;

    @Positive
        public void writeInt(int v) throws IOException;

    @Positive
        public void writeFloat(float v) throws IOException;

    @Positive
        public void writeLong(long v) throws IOException;

    @Positive
        public void writeDouble(double v) throws IOException;

    @Positive
        public void writeBytes(String s) throws IOException;

    @Positive
        public void writeChars(String s) throws IOException;

    @Positive
        public void writeUTF(String s) throws IOException;

    @Positive
        void writeBooleans(boolean[] v, int off, int len) throws IOException;

    @Positive
        void writeChars(char[] v, int off, int len) throws IOException;

    @Positive
        void writeShorts(short[] v, int off, int len) throws IOException;

    @Positive
        void writeInts(int[] v, int off, int len) throws IOException;

    @Positive
        void writeFloats(float[] v, int off, int len) throws IOException;

    @Positive
        void writeLongs(long[] v, int off, int len) throws IOException;

    @Positive
        void writeDoubles(double[] v, int off, int len) throws IOException;

    @Positive
        long getUTFLength(String s);

    @Positive
        void writeUTF(String s, long utflen) throws IOException;

    @Positive
        void writeLongUTF(String s) throws IOException;

    @Positive
        void writeLongUTF(String s, long utflen) throws IOException;
    @Positive
    }

    @Positive
    private static class HandleTable {

    @Positive
        int assign(Object obj);

    @Positive
        int lookup(Object obj);

    @Positive
        void clear();

    @Positive
        int size();
    @Positive
    }

    @Positive
    private static class ReplaceTable {

    @Positive
        void assign(Object obj, Object rep);

    @Positive
        Object lookup(Object obj);

    @Positive
        void clear();

    @Positive
        int size();
    @Positive
    }

    @Positive
    private static class DebugTraceInfoStack {

    @Positive
        void clear();

    @Positive
        void pop();

    @Positive
        void push(String entry);

    @Positive
        public String toString();
    @Positive
    }
    @Positive
}
