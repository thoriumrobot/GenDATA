/*
    @Positive
 * Copyright (c) 2007, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.nio.file;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.ReleasesNoLocks;
    @Positive
import org.checkerframework.checker.mustcall.qual.MustCall;
    @Positive
import org.checkerframework.checker.nonempty.qual.EnsuresNonEmptyIf;
    @Positive
import org.checkerframework.checker.nonempty.qual.NonEmpty;
    @Positive
import org.checkerframework.checker.signedness.qual.PolySigned;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectsOnly;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.BufferedReader;
    @Positive
import java.io.BufferedWriter;
    @Positive
import java.io.Closeable;
    @Positive
import java.io.File;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.InputStreamReader;
    @Positive
import java.io.OutputStream;
    @Positive
import java.io.OutputStreamWriter;
    @Positive
import java.io.Reader;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.io.Writer;
    @Positive
import java.nio.channels.Channels;
    @Positive
import java.nio.channels.FileChannel;
    @Positive
import java.nio.channels.SeekableByteChannel;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.charset.CharsetDecoder;
    @Positive
import java.nio.charset.CharsetEncoder;
    @Positive
import java.nio.charset.StandardCharsets;
    @Positive
import java.nio.file.attribute.BasicFileAttributeView;
    @Positive
import java.nio.file.attribute.BasicFileAttributes;
    @Positive
import java.nio.file.attribute.DosFileAttributes;
    @Positive
import java.nio.file.attribute.FileAttribute;
    @Positive
import java.nio.file.attribute.FileAttributeView;
    @Positive
import java.nio.file.attribute.FileOwnerAttributeView;
    @Positive
import java.nio.file.attribute.FileStoreAttributeView;
    @Positive
import java.nio.file.attribute.FileTime;
    @Positive
import java.nio.file.attribute.PosixFileAttributeView;
    @Positive
import java.nio.file.attribute.PosixFileAttributes;
    @Positive
import java.nio.file.attribute.PosixFilePermission;
    @Positive
import java.nio.file.attribute.UserPrincipal;
    @Positive
import java.nio.file.spi.FileSystemProvider;
    @Positive
import java.nio.file.spi.FileTypeDetector;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collections;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.ServiceLoader;
    @Positive
import java.util.Set;
    @Positive
import java.util.Spliterator;
    @Positive
import java.util.Spliterators;
    @Positive
import java.util.function.BiPredicate;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.stream.StreamSupport;
    @Positive
import jdk.internal.util.ArraysSupport;
    @Positive
import sun.nio.ch.FileChannelImpl;
    @Positive
import sun.nio.cs.UTF_8;
    @Positive
import sun.nio.fs.AbstractFileSystemProvider;

    @Positive
@AnnotatedFor({ "interning", "mustcall", "signedness" })
    @Positive
@UsesObjectEquals
    @Positive
public final class Files {

    @Positive
    @ReleasesNoLocks
    @Positive
    public static InputStream newInputStream(Path path, OpenOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static OutputStream newOutputStream(Path path, OpenOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static SeekableByteChannel newByteChannel(Path path, Set<? extends OpenOption> options, FileAttribute<?>... attrs) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static SeekableByteChannel newByteChannel(Path path, OpenOption... options) throws IOException;

    @Positive
    private static class AcceptAllFilter implements DirectoryStream.Filter<Path> {

    @Positive
        @Override
    @Positive
        @ReleasesNoLocks
    @Positive
        public boolean accept(Path entry);
    @Positive
    }

    @Positive
    @ReleasesNoLocks
    @Positive
    @MustCall("close")
    @Positive
    public static DirectoryStream<Path> newDirectoryStream(Path dir) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    @MustCall("close")
    @Positive
    public static DirectoryStream<Path> newDirectoryStream(Path dir, String glob) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    @MustCall("close")
    @Positive
    public static DirectoryStream<Path> newDirectoryStream(Path dir, DirectoryStream.Filter<? super Path> filter) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path createFile(Path path, FileAttribute<?>... attrs) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path createDirectory(Path dir, FileAttribute<?>... attrs) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path createDirectories(Path dir, FileAttribute<?>... attrs) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path createTempFile(Path dir, String prefix, String suffix, FileAttribute<?>... attrs) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path createTempFile(String prefix, String suffix, FileAttribute<?>... attrs) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path createTempDirectory(Path dir, String prefix, FileAttribute<?>... attrs) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path createTempDirectory(String prefix, FileAttribute<?>... attrs) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path createSymbolicLink(Path link, Path target, FileAttribute<?>... attrs) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path createLink(Path link, Path existing) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static void delete(Path path) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static boolean deleteIfExists(Path path) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path copy(Path source, Path target, CopyOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path move(Path source, Path target, CopyOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path readSymbolicLink(Path link) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static FileStore getFileStore(Path path) throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    public static boolean isSameFile(Path path, Path path2) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static long mismatch(Path path, Path path2) throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    public static boolean isHidden(Path path) throws IOException;

    @Positive
    private static class FileTypeDetectors {
    @Positive
    }

    @Positive
    @ReleasesNoLocks
    @Positive
    public static String probeContentType(Path path) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static <V extends FileAttributeView> V getFileAttributeView(Path path, Class<V> type, LinkOption... options);

    @Positive
    @ReleasesNoLocks
    @Positive
    public static <A extends BasicFileAttributes> A readAttributes(Path path, Class<A> type, LinkOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path setAttribute(Path path, String attribute, Object value, LinkOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Object getAttribute(Path path, String attribute, LinkOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Map<String, Object> readAttributes(Path path, String attributes, LinkOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Set<PosixFilePermission> getPosixFilePermissions(Path path, LinkOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path setPosixFilePermissions(Path path, Set<PosixFilePermission> perms) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static UserPrincipal getOwner(Path path, LinkOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path setOwner(Path path, UserPrincipal owner) throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    public static boolean isSymbolicLink(Path path);

    @Positive
    @SideEffectFree
    @Positive
    public static boolean isDirectory(Path path, LinkOption... options);

    @Positive
    @SideEffectFree
    @Positive
    public static boolean isRegularFile(Path path, LinkOption... options);

    @Positive
    @ReleasesNoLocks
    @Positive
    public static FileTime getLastModifiedTime(Path path, LinkOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path setLastModifiedTime(Path path, FileTime time) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static long size(Path path) throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    public static boolean exists(Path path, LinkOption... options);

    @Positive
    @SideEffectFree
    @Positive
    public static boolean notExists(Path path, LinkOption... options);

    @Positive
    @SideEffectFree
    @Positive
    public static boolean isReadable(Path path);

    @Positive
    @SideEffectFree
    @Positive
    public static boolean isWritable(Path path);

    @Positive
    @SideEffectFree
    @Positive
    public static boolean isExecutable(Path path);

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path walkFileTree(Path start, Set<FileVisitOption> options, int maxDepth, FileVisitor<? super Path> visitor) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path walkFileTree(Path start, FileVisitor<? super Path> visitor) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static BufferedReader newBufferedReader(Path path, Charset cs) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static BufferedReader newBufferedReader(Path path) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static BufferedWriter newBufferedWriter(Path path, Charset cs, OpenOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static BufferedWriter newBufferedWriter(Path path, OpenOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static long copy(InputStream in, Path target, CopyOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static long copy(Path source, OutputStream out) throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    public static byte[] readAllBytes(Path path) throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    public static String readString(Path path) throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    public static String readString(Path path, Charset cs) throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    public static List<String> readAllLines(Path path, Charset cs) throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    public static List<String> readAllLines(Path path) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path write(Path path, @PolySigned byte[] bytes, OpenOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path write(Path path, Iterable<? extends CharSequence> lines, Charset cs, OpenOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path write(Path path, Iterable<? extends CharSequence> lines, OpenOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path writeString(Path path, CharSequence csq, OpenOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    public static Path writeString(Path path, CharSequence csq, Charset cs, OpenOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    @MustCall("close")
    @Positive
    public static Stream<Path> list(Path dir) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    @MustCall("close")
    @Positive
    public static Stream<Path> walk(Path start, int maxDepth, FileVisitOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    @MustCall("close")
    @Positive
    public static Stream<Path> walk(Path start, FileVisitOption... options) throws IOException;

    @Positive
    @ReleasesNoLocks
    @Positive
    @MustCall("close")
    @Positive
    public static Stream<Path> find(Path start, int maxDepth, BiPredicate<Path, BasicFileAttributes> matcher, FileVisitOption... options) throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    @MustCall("close")
    @Positive
    public static Stream<String> lines(Path path, Charset cs) throws IOException;

    @Positive
    @SideEffectFree
    @Positive
    @MustCall("close")
    @Positive
    public static Stream<String> lines(Path path) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
