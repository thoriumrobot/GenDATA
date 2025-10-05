/*
    @Positive
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util.zip;

    @Positive
import org.checkerframework.checker.index.qual.GTENegativeOne;
    @Positive
import org.checkerframework.checker.index.qual.IndexOrHigh;
    @Positive
import org.checkerframework.checker.index.qual.LTEqLengthOf;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signedness.qual.SignedPositive;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.io.Closeable;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.IOException;
    @Positive
import java.io.EOFException;
    @Positive
import java.io.File;
    @Positive
import java.io.RandomAccessFile;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.lang.ref.Cleaner.Cleanable;
    @Positive
import java.nio.charset.CharacterCodingException;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.file.InvalidPathException;
    @Positive
import java.nio.file.attribute.BasicFileAttributes;
    @Positive
import java.nio.file.Files;
    @Positive
import java.util.ArrayDeque;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Deque;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.Objects;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Set;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.TreeSet;
    @Positive
import java.util.WeakHashMap;
    @Positive
import java.util.function.Consumer;
    @Positive
import java.util.function.IntFunction;
    @Positive
import java.util.jar.JarEntry;
    @Positive
import java.util.jar.JarFile;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;
    @Positive
import jdk.internal.access.JavaUtilZipFileAccess;
    @Positive
import jdk.internal.access.JavaUtilJarAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.misc.VM;
    @Positive
import jdk.internal.perf.PerfCounter;
    @Positive
import jdk.internal.ref.CleanerFactory;
    @Positive
import jdk.internal.vm.annotation.Stable;
    @Positive
import sun.nio.cs.UTF_8;
    @Positive
import sun.security.util.SignatureFileVerifier;
    @Positive
import static java.util.zip.ZipConstants64.*;
    @Positive
import static java.util.zip.ZipUtils.*;

    @Positive
@AnnotatedFor({ "index", "interning", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class ZipFile implements ZipConstants, Closeable {

    @Positive
    @SignedPositive
    @Positive
    public static final int OPEN_READ;

    @Positive
    @SignedPositive
    @Positive
    public static final int OPEN_DELETE;

    @Positive
    public ZipFile(String name) throws IOException {
    @Positive
    }

    @Positive
    public ZipFile(File file, int mode) throws IOException {
    @Positive
    }

    @Positive
    public ZipFile(File file) throws ZipException, IOException {
    @Positive
    }

    @Positive
    public ZipFile(File file, int mode, Charset charset) throws IOException {
    @Positive
    }

    @Positive
    public ZipFile(String name, Charset charset) throws IOException {
    @Positive
    }

    @Positive
    public ZipFile(File file, Charset charset) throws IOException {
    @Positive
    }

    @Positive
    @Nullable
    @Positive
    public String getComment();

    @Positive
    @Nullable
    @Positive
    public ZipEntry getEntry(String name);

    @Positive
    @CFComment({ "These @MustCallAlias annotations might not be right.  The", "Javadoc documentation above is not clear.  It seems that closing the", "ZipEntry does close the InputStream, but it is not clear that closing", "the InputStream also closes the ZipEntry." })
    @Positive
    @Nullable
    @Positive
    @MustCallAlias
    @Positive
    public InputStream getInputStream(@MustCallAlias ZipEntry entry) throws IOException;

    @Positive
    private static class InflaterCleanupAction implements Runnable {

    @Positive
        @Override
    @Positive
        public void run();
    @Positive
    }

    @Positive
    private class ZipFileInflaterInputStream extends InflaterInputStream {

    @Positive
        public void close() throws IOException;

    @Positive
        protected void fill() throws IOException;

    @Positive
        public int available() throws IOException;
    @Positive
    }

    @Positive
    public String getName();

    @Positive
    private class ZipEntryIterator<T extends ZipEntry> implements Enumeration<T>, Iterator<T> {

    @Positive
        public ZipEntryIterator(int entryCount) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasMoreElements();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        @EnsuresNonEmptyIf(result = true, expression = "this")
    @Positive
        public boolean hasNext();

    @Positive
        @Override
    @Positive
        @SideEffectsOnly("this")
    @Positive
        public T nextElement(@NonEmpty ZipEntryIterator<T> this);

    @Positive
        @Override
    @Positive
        @SuppressWarnings("unchecked")
    @Positive
        @SideEffectsOnly("this")
    @Positive
        public T next(@NonEmpty ZipEntryIterator<T> this);

    @Positive
        @Override
    @Positive
        public Iterator<T> asIterator();
    @Positive
    }

    @Positive
    public Enumeration<? extends ZipEntry> entries();

    @Positive
    private class EntrySpliterator<T> extends Spliterators.AbstractSpliterator<T> {

    @Positive
        @Override
    @Positive
        public boolean tryAdvance(Consumer<? super T> action);
    @Positive
    }

    @Positive
    public Stream<? extends ZipEntry> stream();

    @Positive
    @Pure
    @Positive
    @NonNegative
    @Positive
    public int size();

    @Positive
    private static class CleanableResource implements Runnable {

    @Positive
        void clean();

    @Positive
        Inflater getInflater();

    @Positive
        void releaseInflater(Inflater inf);

    @Positive
        public void run();
    @Positive
    }

    @Positive
    public void close() throws IOException;

    @Positive
    private class ZipFileInputStream extends InputStream {

    @Positive
        protected long rem;

    @Positive
        protected long size;

    @Positive
        @GTENegativeOne
    @Positive
        @LTEqLengthOf({ "#1" })
    @Positive
        public int read(byte[] b, @IndexOrHigh({ "#1" }) int off, @IndexOrHigh({ "#1" }) int len) throws IOException;

    @Positive
        public int read() throws IOException;

    @Positive
        public long skip(long n) throws IOException;

    @Positive
        public int available();

    @Positive
        public long size();

    @Positive
        public void close();
    @Positive
    }

    @Positive
    private static class Source {

    @Positive
        private static class Key {

    @Positive
            public Key(File file, BasicFileAttributes attrs, ZipCoder zc) {
    @Positive
            }

    @Positive
            public int hashCode();

    @Positive
            public boolean equals(Object obj);
    @Positive
        }

    @Positive
        static Source get(File file, boolean toDelete, ZipCoder zc) throws IOException;

    @Positive
        static void release(Source src) throws IOException;

    @Positive
        private static class End {
    @Positive
        }
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
