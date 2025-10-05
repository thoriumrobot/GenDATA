/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util.zip;

import org.checkerframework.checker.index.qual.GTENegativeOne;
import org.checkerframework.checker.index.qual.IndexOrHigh;
import org.checkerframework.checker.index.qual.LTEqLengthOf;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.mustcall.qual.MustCallAlias;
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
import org.checkerframework.checker.nonempty.qual.NonEmpty;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signedness.qual.SignedPositive;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectsOnly;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import java.io.Closeable;
import java.io.InputStream;
import java.io.IOException;
import java.io.EOFException;
import java.io.File;
import java.io.RandomAccessFile;
import java.io.UncheckedIOException;
import java.lang.ref.Cleaner.Cleanable;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.Charset;
import java.nio.file.InvalidPathException;
import java.nio.file.attribute.BasicFileAttributes;
import java.nio.file.Files;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.TreeSet;
import java.util.WeakHashMap;
import java.util.function.Consumer;
import java.util.function.IntFunction;
import java.util.jar.JarEntry;
import java.util.jar.JarFile;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;
import jdk.internal.access.JavaUtilZipFileAccess;
import jdk.internal.access.JavaUtilJarAccess;
import jdk.internal.access.SharedSecrets;
import jdk.internal.misc.VM;
import jdk.internal.perf.PerfCounter;
import jdk.internal.ref.CleanerFactory;
import jdk.internal.vm.annotation.Stable;
import sun.nio.cs.UTF_8;
import sun.security.util.SignatureFileVerifier;
import static java.util.zip.ZipConstants64.*;
import static java.util.zip.ZipUtils.*;

@AnnotatedFor({ "index", "interning", "nullness" })
@UsesObjectEquals
public class ZipFile implements ZipConstants, Closeable {

    @SignedPositive
    public static final int OPEN_READ;

    @SignedPositive
    public static final int OPEN_DELETE;

    public ZipFile(String name) throws IOException {
    }

    public ZipFile(File file, int mode) throws IOException {
    }

    public ZipFile(File file) throws ZipException, IOException {
    }

    public ZipFile(File file, int mode, Charset charset) throws IOException {
    }

    public ZipFile(String name, Charset charset) throws IOException {
    }

    public ZipFile(File file, Charset charset) throws IOException {
    }

    @Nullable
    public String getComment();

    @Nullable
    public ZipEntry getEntry(String name);

    @CFComment({ "These @MustCallAlias annotations might not be right.  The", "Javadoc documentation above is not clear.  It seems that closing the", "ZipEntry does close the InputStream, but it is not clear that closing", "the InputStream also closes the ZipEntry." })
    @Nullable
    @MustCallAlias
    public InputStream getInputStream(@MustCallAlias ZipEntry entry) throws IOException;

    private static class InflaterCleanupAction implements Runnable {

        @Override
        public void run();
    }

    private class ZipFileInflaterInputStream extends InflaterInputStream {

        public void close() throws IOException;

        protected void fill() throws IOException;

        public int available() throws IOException;
    }

    public String getName();

    private class ZipEntryIterator<T extends ZipEntry> implements Enumeration<T>, Iterator<T> {

        public ZipEntryIterator(int entryCount) {
        }

        @Override
        @EnsuresNonEmptyIf(result = true, expression = "this")
        public boolean hasMoreElements();

        @Override
        @Pure
        @EnsuresNonEmptyIf(result = true, expression = "this")
        public boolean hasNext();

        @Override
        @SideEffectsOnly("this")
        public T nextElement(@NonEmpty ZipEntryIterator<T> this);

        @Override
        @SuppressWarnings("unchecked")
        @SideEffectsOnly("this")
        public T next(@NonEmpty ZipEntryIterator<T> this);

        @Override
        public Iterator<T> asIterator();
    }

    public Enumeration<? extends ZipEntry> entries();

    private class EntrySpliterator<T> extends Spliterators.AbstractSpliterator<T> {

        @Override
        public boolean tryAdvance(Consumer<? super T> action);
    }

    public Stream<? extends ZipEntry> stream();

    @Pure
    @NonNegative
    public int size();

    private static class CleanableResource implements Runnable {

        void clean();

        Inflater getInflater();

        void releaseInflater(Inflater inf);

        public void run();
    }

    public void close() throws IOException;

    private class ZipFileInputStream extends InputStream {

        protected long rem;

        protected long size;

        @GTENegativeOne
        @LTEqLengthOf({ "#1" })
        public int read(byte[] b, @IndexOrHigh({ "#1" }) int off, @IndexOrHigh({ "#1" }) int len) throws IOException;

        public int read() throws IOException;

        public long skip(long n) throws IOException;

        public int available();

        public long size();

        public void close();
    }

    private static class Source {

        private static class Key {

            public Key(File file, BasicFileAttributes attrs, ZipCoder zc) {
            }

            public int hashCode();

            public boolean equals(Object obj);
        }

        static Source get(File file, boolean toDelete, ZipCoder zc) throws IOException;

        static void release(Source src) throws IOException;

        private static class End {
        }
    }
}
