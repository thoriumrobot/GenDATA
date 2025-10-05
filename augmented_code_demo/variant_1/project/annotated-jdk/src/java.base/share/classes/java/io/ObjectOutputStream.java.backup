/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.io;

import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signedness.qual.PolySigned;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.io.ObjectStreamClass.WeakClassKey;
import java.lang.ref.ReferenceQueue;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.StringJoiner;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import static java.io.ObjectStreamClass.processQueue;
import sun.reflect.misc.ReflectUtil;

@AnnotatedFor({ "nullness", "index", "signedness" })
public class ObjectOutputStream extends OutputStream implements ObjectOutput, ObjectStreamConstants {

    private static class Caches {
    }

    @MustCallAlias
    public ObjectOutputStream(@MustCallAlias OutputStream out) throws IOException {
    }

    protected ObjectOutputStream() throws IOException, SecurityException {
    }

    public void useProtocolVersion(int version) throws IOException;

    public final void writeObject(@Nullable Object obj) throws IOException;

    protected void writeObjectOverride(Object obj) throws IOException;

    public void writeUnshared(@Nullable Object obj) throws IOException;

    public void defaultWriteObject() throws IOException;

    public ObjectOutputStream.PutField putFields() throws IOException;

    public void writeFields() throws IOException;

    public void reset() throws IOException;

    protected void annotateClass(Class<?> cl) throws IOException;

    protected void annotateProxyClass(Class<?> cl) throws IOException;

    protected Object replaceObject(Object obj) throws IOException;

    protected boolean enableReplaceObject(boolean enable) throws SecurityException;

    protected void writeStreamHeader() throws IOException;

    protected void writeClassDescriptor(ObjectStreamClass desc) throws IOException;

    public void write(@PolySigned int val) throws IOException;

    public void write(@PolySigned byte[] buf) throws IOException;

    public void write(@PolySigned byte[] buf, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    public void flush() throws IOException;

    protected void drain() throws IOException;

    public void close() throws IOException;

    public void writeBoolean(boolean val) throws IOException;

    public void writeByte(int val) throws IOException;

    public void writeShort(int val) throws IOException;

    public void writeChar(int val) throws IOException;

    public void writeInt(int val) throws IOException;

    public void writeLong(long val) throws IOException;

    public void writeFloat(float val) throws IOException;

    public void writeDouble(double val) throws IOException;

    public void writeBytes(String str) throws IOException;

    public void writeChars(String str) throws IOException;

    public void writeUTF(String str) throws IOException;

    public abstract static class PutField {

        public PutField() {
        }

        public abstract void put(String name, boolean val);

        public abstract void put(String name, byte val);

        public abstract void put(String name, char val);

        public abstract void put(String name, short val);

        public abstract void put(String name, int val);

        public abstract void put(String name, long val);

        public abstract void put(String name, float val);

        public abstract void put(String name, double val);

        public abstract void put(String name, @Nullable Object val);

        @Deprecated
        public abstract void write(ObjectOutput out) throws IOException;
    }

    int getProtocolVersion();

    void writeTypeString(String str) throws IOException;

    private class PutFieldImpl extends PutField {

        public void put(String name, boolean val);

        public void put(String name, byte val);

        public void put(String name, char val);

        public void put(String name, short val);

        public void put(String name, int val);

        public void put(String name, float val);

        public void put(String name, long val);

        public void put(String name, double val);

        public void put(String name, Object val);

        public void write(ObjectOutput out) throws IOException;

        void writeFields() throws IOException;
    }

    private static class BlockDataOutputStream extends OutputStream implements DataOutput {

        boolean setBlockDataMode(boolean mode) throws IOException;

        boolean getBlockDataMode();

        public void write(int b) throws IOException;

        public void write(byte[] b) throws IOException;

        public void write(byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

        public void flush() throws IOException;

        public void close() throws IOException;

        void write(byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len, boolean copy) throws IOException;

        void drain() throws IOException;

        public void writeBoolean(boolean v) throws IOException;

        public void writeByte(int v) throws IOException;

        public void writeChar(int v) throws IOException;

        public void writeShort(int v) throws IOException;

        public void writeInt(int v) throws IOException;

        public void writeFloat(float v) throws IOException;

        public void writeLong(long v) throws IOException;

        public void writeDouble(double v) throws IOException;

        public void writeBytes(String s) throws IOException;

        public void writeChars(String s) throws IOException;

        public void writeUTF(String s) throws IOException;

        void writeBooleans(boolean[] v, int off, int len) throws IOException;

        void writeChars(char[] v, int off, int len) throws IOException;

        void writeShorts(short[] v, int off, int len) throws IOException;

        void writeInts(int[] v, int off, int len) throws IOException;

        void writeFloats(float[] v, int off, int len) throws IOException;

        void writeLongs(long[] v, int off, int len) throws IOException;

        void writeDoubles(double[] v, int off, int len) throws IOException;

        long getUTFLength(String s);

        void writeUTF(String s, long utflen) throws IOException;

        void writeLongUTF(String s) throws IOException;

        void writeLongUTF(String s, long utflen) throws IOException;
    }

    private static class HandleTable {

        int assign(Object obj);

        int lookup(Object obj);

        void clear();

        int size();
    }

    private static class ReplaceTable {

        void assign(Object obj, Object rep);

        Object lookup(Object obj);

        void clear();

        int size();
    }

    private static class DebugTraceInfoStack {

        void clear();

        void pop();

        void push(String entry);

        public String toString();
    }
}
