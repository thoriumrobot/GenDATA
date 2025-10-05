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
import org.checkerframework.checker.signedness.qual.SignedPositive;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.ObjectInputFilter.Config;
    @Positive
import java.io.ObjectStreamClass.WeakClassKey;
    @Positive
import java.io.ObjectStreamClass.RecordSupport;
    @Positive
import java.lang.System.Logger;
    @Positive
import java.lang.invoke.MethodHandle;
    @Positive
import java.lang.ref.ReferenceQueue;
    @Positive
import java.lang.reflect.Array;
    @Positive
import java.lang.reflect.InvocationHandler;
    @Positive
import java.lang.reflect.Modifier;
    @Positive
import java.lang.reflect.Proxy;
    @Positive
import java.security.AccessControlContext;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.PrivilegedActionException;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import static java.io.ObjectStreamClass.processQueue;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.event.DeserializationEvent;
    @Positive
import jdk.internal.misc.Unsafe;
    @Positive
import sun.reflect.misc.ReflectUtil;
    @Positive
import sun.security.action.GetBooleanAction;
    @Positive
import sun.security.action.GetIntegerAction;

    @Positive
@AnnotatedFor({ "nullness", "index", "signedness" })
    @Positive
public class ObjectInputStream extends InputStream implements ObjectInput, ObjectStreamConstants {

    @Positive
    private static class Caches {
    @Positive
    }

    @Positive
    private static class Logging {
    @Positive
    }

    @Positive
    @MustCallAlias
    @Positive
    public ObjectInputStream(@MustCallAlias InputStream in) throws IOException {
    @Positive
    }

    @Positive
    protected ObjectInputStream() throws IOException, SecurityException {
    @Positive
    }

    @Positive
    public final Object readObject() throws IOException, ClassNotFoundException;

    @Positive
    protected Object readObjectOverride() throws IOException, ClassNotFoundException;

    @Positive
    public Object readUnshared() throws IOException, ClassNotFoundException;

    @Positive
    public void defaultReadObject() throws IOException, ClassNotFoundException;

    @Positive
    public ObjectInputStream.GetField readFields() throws IOException, ClassNotFoundException;

    @Positive
    public void registerValidation(ObjectInputValidation obj, int prio) throws NotActiveException, InvalidObjectException;

    @Positive
    protected Class<?> resolveClass(ObjectStreamClass desc) throws IOException, ClassNotFoundException;

    @Positive
    protected Class<?> resolveProxyClass(String[] interfaces) throws IOException, ClassNotFoundException;

    @Positive
    protected Object resolveObject(Object obj) throws IOException;

    @Positive
    protected boolean enableResolveObject(boolean enable) throws SecurityException;

    @Positive
    protected void readStreamHeader() throws IOException, StreamCorruptedException;

    @Positive
    protected ObjectStreamClass readClassDescriptor() throws IOException, ClassNotFoundException;

    @Positive
    public int read() throws IOException;

    @Positive
    @GTENegativeOne
    @Positive
    @LTEqLengthOf({ "#1" })
    @Positive
    public int read(byte[] buf, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
    @NonNegative
    @Positive
    public int available() throws IOException;

    @Positive
    public void close() throws IOException;

    @Positive
    public boolean readBoolean() throws IOException;

    @Positive
    public byte readByte() throws IOException;

    @Positive
    @SignedPositive
    @Positive
    @NonNegative
    @Positive
    public int readUnsignedByte() throws IOException;

    @Positive
    public char readChar() throws IOException;

    @Positive
    public short readShort() throws IOException;

    @Positive
    @SignedPositive
    @Positive
    @NonNegative
    @Positive
    public int readUnsignedShort() throws IOException;

    @Positive
    public int readInt() throws IOException;

    @Positive
    public long readLong() throws IOException;

    @Positive
    public float readFloat() throws IOException;

    @Positive
    public double readDouble() throws IOException;

    @Positive
    public void readFully(byte[] buf) throws IOException;

    @Positive
    public void readFully(byte[] buf, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
    @NonNegative
    @Positive
    public int skipBytes(@NonNegative int len) throws IOException;

    @Positive
    @Deprecated
    @Positive
    @Nullable
    @Positive
    public String readLine() throws IOException;

    @Positive
    public String readUTF() throws IOException;

    @Positive
    public final ObjectInputFilter getObjectInputFilter();

    @Positive
    public final void setObjectInputFilter(ObjectInputFilter filter);

    @Positive
    public abstract static class GetField {

    @Positive
        public GetField() {
    @Positive
        }

    @Positive
        public abstract ObjectStreamClass getObjectStreamClass();

    @Positive
        public abstract boolean defaulted(String name) throws IOException;

    @Positive
        public abstract boolean get(String name, boolean val) throws IOException;

    @Positive
        public abstract byte get(String name, byte val) throws IOException;

    @Positive
        public abstract char get(String name, char val) throws IOException;

    @Positive
        public abstract short get(String name, short val) throws IOException;

    @Positive
        public abstract int get(String name, int val) throws IOException;

    @Positive
        public abstract long get(String name, long val) throws IOException;

    @Positive
        public abstract float get(String name, float val) throws IOException;

    @Positive
        public abstract double get(String name, double val) throws IOException;

    @Positive
        @Nullable
    @Positive
        public abstract Object get(String name, @Nullable Object val) throws IOException;
    @Positive
    }

    @Positive
    String readTypeString() throws IOException;

    @Positive
    private final class FieldValues extends GetField {

    @Positive
        public ObjectStreamClass getObjectStreamClass();

    @Positive
        public boolean defaulted(String name);

    @Positive
        public boolean get(String name, boolean val);

    @Positive
        public byte get(String name, byte val);

    @Positive
        public char get(String name, char val);

    @Positive
        public short get(String name, short val);

    @Positive
        public int get(String name, int val);

    @Positive
        public float get(String name, float val);

    @Positive
        public long get(String name, long val);

    @Positive
        public double get(String name, double val);

    @Positive
        public Object get(String name, Object val);

    @Positive
        void defaultCheckFieldValues(Object obj);
    @Positive
    }

    @Positive
    private static class ValidationList {

    @Positive
        private static class Callback {
    @Positive
        }

    @Positive
        void register(ObjectInputValidation obj, int priority) throws InvalidObjectException;

    @Positive
        @SuppressWarnings("removal")
    @Positive
        void doCallbacks() throws InvalidObjectException;

    @Positive
        public void clear();
    @Positive
    }

    @Positive
    static class FilterValues implements ObjectInputFilter.FilterInfo {

    @Positive
        public FilterValues(Class<?> clazz, long arrayLength, long totalObjectRefs, long depth, long streamBytes) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public Class<?> serialClass();

    @Positive
        @Override
    @Positive
        public long arrayLength();

    @Positive
        @Override
    @Positive
        public long references();

    @Positive
        @Override
    @Positive
        public long depth();

    @Positive
        @Override
    @Positive
        public long streamBytes();
    @Positive
    }

    @Positive
    private static class PeekInputStream extends InputStream {

    @Positive
        @Pure
    @Positive
        int peek() throws IOException;

    @Positive
        public int read() throws IOException;

    @Positive
        public int read(byte[] b, int off, int len) throws IOException;

    @Positive
        void readFully(byte[] b, int off, int len) throws IOException;

    @Positive
        public long skip(long n) throws IOException;

    @Positive
        public int available() throws IOException;

    @Positive
        public void close() throws IOException;

    @Positive
        public long getBytesRead();
    @Positive
    }

    @Positive
    private class BlockDataInputStream extends InputStream implements DataInput {

    @Positive
        boolean setBlockDataMode(boolean newmode) throws IOException;

    @Positive
        boolean getBlockDataMode();

    @Positive
        void skipBlockData() throws IOException;

    @Positive
        int currentBlockRemaining();

    @Positive
        @Pure
    @Positive
        int peek() throws IOException;

    @Positive
        @Pure
    @Positive
        byte peekByte() throws IOException;

    @Positive
        public int read() throws IOException;

    @Positive
        public int read(byte[] b, int off, int len) throws IOException;

    @Positive
        public long skip(long len) throws IOException;

    @Positive
        public int available() throws IOException;

    @Positive
        public void close() throws IOException;

    @Positive
        int read(byte[] b, int off, int len, boolean copy) throws IOException;

    @Positive
        public void readFully(byte[] b) throws IOException;

    @Positive
        public void readFully(byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len) throws IOException;

    @Positive
        public void readFully(byte[] b, @IndexOrHigh({ "#1" }) int off, @LTLengthOf(value = { "#1" }, offset = { "#2 - 1" }) @NonNegative int len, boolean copy) throws IOException;

    @Positive
        public int skipBytes(int n) throws IOException;

    @Positive
        public boolean readBoolean() throws IOException;

    @Positive
        public byte readByte() throws IOException;

    @Positive
        @SignedPositive
    @Positive
        @NonNegative
    @Positive
        public int readUnsignedByte() throws IOException;

    @Positive
        public char readChar() throws IOException;

    @Positive
        public short readShort() throws IOException;

    @Positive
        @SignedPositive
    @Positive
        @NonNegative
    @Positive
        public int readUnsignedShort() throws IOException;

    @Positive
        public int readInt() throws IOException;

    @Positive
        public float readFloat() throws IOException;

    @Positive
        public long readLong() throws IOException;

    @Positive
        public double readDouble() throws IOException;

    @Positive
        public String readUTF() throws IOException;

    @Positive
        @SuppressWarnings("deprecation")
    @Positive
        public String readLine() throws IOException;

    @Positive
        void readBooleans(boolean[] v, int off, int len) throws IOException;

    @Positive
        void readChars(char[] v, int off, int len) throws IOException;

    @Positive
        void readShorts(short[] v, int off, int len) throws IOException;

    @Positive
        void readInts(int[] v, int off, int len) throws IOException;

    @Positive
        void readFloats(float[] v, int off, int len) throws IOException;

    @Positive
        void readLongs(long[] v, int off, int len) throws IOException;

    @Positive
        void readDoubles(double[] v, int off, int len) throws IOException;

    @Positive
        String readLongUTF() throws IOException;

    @Positive
        long getBytesRead();
    @Positive
    }

    @Positive
    private static class HandleTable {

    @Positive
        int assign(Object obj);

    @Positive
        void markDependency(int dependent, int target);

    @Positive
        void markException(int handle, ClassNotFoundException ex);

    @Positive
        void finish(int handle);

    @Positive
        void setObject(int handle, Object obj);

    @Positive
        Object lookupObject(int handle);

    @Positive
        ClassNotFoundException lookupException(int handle);

    @Positive
        void clear();

    @Positive
        int size();

    @Positive
        private static class HandleList {

    @Positive
            public HandleList() {
    @Positive
            }

    @Positive
            public void add(int handle);

    @Positive
            public int get(int index);

    @Positive
            public int size();
    @Positive
        }
    @Positive
    }
    @Positive
}
