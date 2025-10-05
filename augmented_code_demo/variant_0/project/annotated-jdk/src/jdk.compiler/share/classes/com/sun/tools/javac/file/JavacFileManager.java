/*
    @Positive
 * Copyright (c) 2005, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.file;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.File;
    @Positive
import java.io.IOException;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.lang.module.Configuration;
    @Positive
import java.lang.module.ModuleFinder;
    @Positive
import java.net.MalformedURLException;
    @Positive
import java.net.URI;
    @Positive
import java.net.URISyntaxException;
    @Positive
import java.net.URL;
    @Positive
import java.nio.CharBuffer;
    @Positive
import java.nio.charset.Charset;
    @Positive
import java.nio.file.FileSystem;
    @Positive
import java.nio.file.FileSystems;
    @Positive
import java.nio.file.FileVisitOption;
    @Positive
import java.nio.file.FileVisitResult;
    @Positive
import java.nio.file.Files;
    @Positive
import java.nio.file.InvalidPathException;
    @Positive
import java.nio.file.LinkOption;
    @Positive
import java.nio.file.Path;
    @Positive
import java.nio.file.Paths;
    @Positive
import java.nio.file.ProviderNotFoundException;
    @Positive
import java.nio.file.SimpleFileVisitor;
    @Positive
import java.nio.file.attribute.BasicFileAttributes;
    @Positive
import java.nio.file.spi.FileSystemProvider;
    @Positive
import java.util.ArrayList;
    @Positive
import java.util.Arrays;
    @Positive
import java.util.Collection;
    @Positive
import java.util.Collections;
    @Positive
import java.util.Comparator;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.ServiceLoader;
    @Positive
import java.util.Set;
    @Positive
import java.util.stream.Collectors;
    @Positive
import java.util.stream.Stream;
    @Positive
import javax.lang.model.SourceVersion;
    @Positive
import javax.tools.FileObject;
    @Positive
import javax.tools.JavaFileManager;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import javax.tools.StandardJavaFileManager;
    @Positive
import com.sun.tools.javac.file.RelativePath.RelativeDirectory;
    @Positive
import com.sun.tools.javac.file.RelativePath.RelativeFile;
    @Positive
import com.sun.tools.javac.main.Option;
    @Positive
import com.sun.tools.javac.resources.CompilerProperties.Errors;
    @Positive
import com.sun.tools.javac.util.Assert;
    @Positive
import com.sun.tools.javac.util.Context;
    @Positive
import com.sun.tools.javac.util.Context.Factory;
    @Positive
import com.sun.tools.javac.util.DefinedBy;
    @Positive
import com.sun.tools.javac.util.DefinedBy.Api;
    @Positive
import com.sun.tools.javac.util.List;
    @Positive
import com.sun.tools.javac.util.ListBuffer;
    @Positive
import static java.nio.file.FileVisitOption.FOLLOW_LINKS;
    @Positive
import static javax.tools.StandardLocation.*;

    @Positive
public class JavacFileManager extends BaseFileManager implements StandardJavaFileManager {

    @Positive
    public static char[] toArray(CharBuffer buffer);

    @Positive
    protected boolean symbolFileEnabled;

    @Positive
    protected enum SortFiles implements Comparator<Path> {

    @Positive
        FORWARD {

    @Positive
            @Override
    @Positive
            public int compare(Path f1, Path f2);
    @Positive
        }
    @Positive
        , REVERSE {

    @Positive
            @Override
    @Positive
            public int compare(Path f1, Path f2);
    @Positive
        }

    @Positive
    }

    @Positive
    protected SortFiles sortFiles;

    @Positive
    public static void preRegister(Context context);

    @Positive
    public JavacFileManager(Context context, boolean register, Charset charset) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public void setContext(Context context);

    @Positive
    @Override
    @Positive
    @DefinedBy(DefinedBy.Api.COMPILER)
    @Positive
    public void setPathFactory(PathFactory f);

    @Positive
    public void setSymbolFileEnabled(boolean b);

    @Positive
    public boolean isSymbolFileEnabled();

    @Positive
    public JavaFileObject getJavaFileObject(String name);

    @Positive
    public JavaFileObject getJavaFileObject(Path file);

    @Positive
    public JavaFileObject getFileForOutput(String classname, JavaFileObject.Kind kind, JavaFileObject sibling) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Iterable<? extends JavaFileObject> getJavaFileObjectsFromStrings(Iterable<String> names);

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Iterable<? extends JavaFileObject> getJavaFileObjects(String... names);

    @Positive
    public static void testName(String name, boolean isValidPackageName, boolean isValidClassName);

    @Positive
    synchronized Container getContainer(Path path) throws IOException;

    @Positive
    private interface Container {

    @Positive
        public abstract void list(Path userPath, RelativeDirectory subdirectory, Set<JavaFileObject.Kind> fileKinds, boolean recurse, ListBuffer<JavaFileObject> resultList) throws IOException;

    @Positive
        public abstract JavaFileObject getFileObject(Path userPath, RelativeFile name) throws IOException;

    @Positive
        public abstract void close() throws IOException;

    @Positive
        public abstract boolean maintainsDirectoryIndex();

    @Positive
        public abstract Iterable<RelativeDirectory> indexedDirectories();
    @Positive
    }

    @Positive
    private final class JRTImageContainer implements Container {

    @Positive
        @Override
    @Positive
        public void list(Path userPath, RelativeDirectory subdirectory, Set<JavaFileObject.Kind> fileKinds, boolean recurse, ListBuffer<JavaFileObject> resultList) throws IOException;

    @Positive
        @Override
    @Positive
        public JavaFileObject getFileObject(Path userPath, RelativeFile name) throws IOException;

    @Positive
        @Override
    @Positive
        public void close() throws IOException;

    @Positive
        @Override
    @Positive
        public boolean maintainsDirectoryIndex();

    @Positive
        @Override
    @Positive
        public Iterable<RelativeDirectory> indexedDirectories();
    @Positive
    }

    @Positive
    private final class DirectoryContainer implements Container {

    @Positive
        public DirectoryContainer(Path directory) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public void list(Path userPath, RelativeDirectory subdirectory, Set<JavaFileObject.Kind> fileKinds, boolean recurse, ListBuffer<JavaFileObject> resultList) throws IOException;

    @Positive
        @Override
    @Positive
        public JavaFileObject getFileObject(Path userPath, RelativeFile name) throws IOException;

    @Positive
        @Override
    @Positive
        public void close() throws IOException;

    @Positive
        @Override
    @Positive
        public boolean maintainsDirectoryIndex();

    @Positive
        @Override
    @Positive
        public Iterable<RelativeDirectory> indexedDirectories();
    @Positive
    }

    @Positive
    private final class ArchiveContainer implements Container {

    @Positive
        public ArchiveContainer(Path archivePath) throws IOException, ProviderNotFoundException, SecurityException {
    @Positive
        }

    @Positive
        @Override
    @Positive
        public void list(Path userPath, RelativeDirectory subdirectory, Set<JavaFileObject.Kind> fileKinds, boolean recurse, ListBuffer<JavaFileObject> resultList) throws IOException;

    @Positive
        @Override
    @Positive
        public JavaFileObject getFileObject(Path userPath, RelativeFile name) throws IOException;

    @Positive
        @Override
    @Positive
        public void close() throws IOException;

    @Positive
        @Override
    @Positive
        public boolean maintainsDirectoryIndex();

    @Positive
        @Override
    @Positive
        public Iterable<RelativeDirectory> indexedDirectories();
    @Positive
    }

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public void flush();

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public void close() throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public ClassLoader getClassLoader(Location location);

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Iterable<JavaFileObject> list(Location location, String packageName, Set<JavaFileObject.Kind> kinds, boolean recurse) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public String inferBinaryName(Location location, JavaFileObject file);

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public boolean isSameFile(FileObject a, FileObject b);

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public boolean hasLocation(Location location);

    @Positive
    protected boolean hasExplicitLocation(Location location);

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public JavaFileObject getJavaFileForInput(Location location, String className, JavaFileObject.Kind kind) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public FileObject getFileForInput(Location location, String packageName, String relativeName) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public JavaFileObject getJavaFileForOutput(Location location, String className, JavaFileObject.Kind kind, FileObject sibling) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public FileObject getFileForOutput(Location location, String packageName, String relativeName, FileObject sibling) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Iterable<? extends JavaFileObject> getJavaFileObjectsFromFiles(Iterable<? extends File> files);

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Iterable<? extends JavaFileObject> getJavaFileObjectsFromPaths(Collection<? extends Path> paths);

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Iterable<? extends JavaFileObject> getJavaFileObjects(File... files);

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Iterable<? extends JavaFileObject> getJavaFileObjects(Path... paths);

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public void setLocation(Location location, Iterable<? extends File> searchpath) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public void setLocationFromPaths(Location location, Collection<? extends Path> searchpath) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Iterable<? extends File> getLocation(Location location);

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Collection<? extends Path> getLocationAsPaths(Location location);

    @Positive
    private static class PathAndContainer implements Comparable<PathAndContainer> {

    @Positive
        @Override
    @Positive
        public int compareTo(PathAndContainer other);

    @Positive
        @Override
    @Positive
        public boolean equals(Object o);

    @Positive
        @Override
    @Positive
        public int hashCode();
    @Positive
    }

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    @Pure
    @Positive
    public boolean contains(Location location, FileObject fo) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Location getLocationForModule(Location location, String moduleName) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public <S> ServiceLoader<S> getServiceLoader(Location location, Class<S> service) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Location getLocationForModule(Location location, JavaFileObject fo) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public void setLocationForModule(Location location, String moduleName, Collection<? extends Path> paths) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public String inferModuleName(Location location);

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Iterable<Set<Location>> listLocationsForModules(Location location) throws IOException;

    @Positive
    @Override
    @Positive
    @DefinedBy(Api.COMPILER)
    @Positive
    public Path asPath(FileObject file);

    @Positive
    protected static boolean isRelativeUri(URI uri);

    @Positive
    protected static boolean isRelativeUri(String u);

    @Positive
    public static String getRelativeName(File file);

    @Positive
    public static String getMessage(IOException e);

    @Positive
    @Override
    @Positive
    public boolean handleOption(Option option, String value);
    @Positive
}

// CFWR semantic augmentation - variant 0
