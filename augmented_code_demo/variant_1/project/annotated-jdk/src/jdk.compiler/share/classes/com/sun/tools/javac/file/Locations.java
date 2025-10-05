/*
    @Positive
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
import java.io.Closeable;
    @Positive
import java.io.File;
    @Positive
import java.io.FileNotFoundException;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.net.URI;
    @Positive
import java.net.URL;
    @Positive
import java.net.URLClassLoader;
    @Positive
import java.nio.file.DirectoryIteratorException;
    @Positive
import java.nio.file.DirectoryStream;
    @Positive
import java.nio.file.FileSystem;
    @Positive
import java.nio.file.FileSystemNotFoundException;
    @Positive
import java.nio.file.FileSystems;
    @Positive
import java.nio.file.Files;
    @Positive
import java.nio.file.InvalidPathException;
    @Positive
import java.nio.file.Path;
    @Positive
import java.nio.file.Paths;
    @Positive
import java.nio.file.ProviderNotFoundException;
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
import java.util.EnumMap;
    @Positive
import java.util.EnumSet;
    @Positive
import java.util.HashMap;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.LinkedHashMap;
    @Positive
import java.util.LinkedHashSet;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Objects;
    @Positive
import java.util.NoSuchElementException;
    @Positive
import java.util.Set;
    @Positive
import java.util.function.Predicate;
    @Positive
import java.util.regex.Matcher;
    @Positive
import java.util.regex.Pattern;
    @Positive
import java.util.stream.Collectors;
    @Positive
import java.util.stream.Stream;
    @Positive
import java.util.jar.Attributes;
    @Positive
import java.util.jar.Manifest;
    @Positive
import javax.lang.model.SourceVersion;
    @Positive
import javax.tools.JavaFileManager;
    @Positive
import javax.tools.JavaFileManager.Location;
    @Positive
import javax.tools.JavaFileObject;
    @Positive
import javax.tools.StandardJavaFileManager;
    @Positive
import javax.tools.StandardJavaFileManager.PathFactory;
    @Positive
import javax.tools.StandardLocation;
    @Positive
import jdk.internal.jmod.JmodFile;
    @Positive
import com.sun.tools.javac.code.Lint;
    @Positive
import com.sun.tools.javac.code.Lint.LintCategory;
    @Positive
import com.sun.tools.javac.main.Option;
    @Positive
import com.sun.tools.javac.resources.CompilerProperties.Errors;
    @Positive
import com.sun.tools.javac.resources.CompilerProperties.Warnings;
    @Positive
import com.sun.tools.javac.util.DefinedBy;
    @Positive
import com.sun.tools.javac.util.DefinedBy.Api;
    @Positive
import com.sun.tools.javac.util.JCDiagnostic.Warning;
    @Positive
import com.sun.tools.javac.util.ListBuffer;
    @Positive
import com.sun.tools.javac.util.Log;
    @Positive
import com.sun.tools.javac.jvm.ModuleNameReader;
    @Positive
import com.sun.tools.javac.util.Iterators;
    @Positive
import com.sun.tools.javac.util.Pair;
    @Positive
import com.sun.tools.javac.util.StringUtils;
    @Positive
import static javax.tools.StandardLocation.SYSTEM_MODULES;
    @Positive
import static javax.tools.StandardLocation.PLATFORM_CLASS_PATH;
    @Positive
import static com.sun.tools.javac.main.Option.BOOT_CLASS_PATH;
    @Positive
import static com.sun.tools.javac.main.Option.ENDORSEDDIRS;
    @Positive
import static com.sun.tools.javac.main.Option.EXTDIRS;
    @Positive
import static com.sun.tools.javac.main.Option.XBOOTCLASSPATH_APPEND;
    @Positive
import static com.sun.tools.javac.main.Option.XBOOTCLASSPATH_PREPEND;

    @Positive
public class Locations {

    @Positive
    Path getPath(String first, String... more);

    @Positive
    public void close() throws IOException;

    @Positive
    void update(Log log, boolean warn, FSInfo fsInfo);

    @Positive
    void setPathFactory(PathFactory f);

    @Positive
    boolean isDefaultBootClassPath();

    @Positive
    boolean isDefaultSystemModulesPath();

    @Positive
    public void setMultiReleaseValue(String multiReleaseValue);

    @Positive
    private class SearchPath extends LinkedHashSet<Path> {

    @Positive
        public SearchPath expandJarClassPaths(boolean x);

    @Positive
        public SearchPath emptyPathDefault(Path x);

    @Positive
        public SearchPath addDirectories(String dirs, boolean warn);

    @Positive
        public SearchPath addDirectories(String dirs);

    @Positive
        public SearchPath addFiles(String files, boolean warn);

    @Positive
        public SearchPath addFiles(String files);

    @Positive
        public SearchPath addFiles(Iterable<? extends Path> files, boolean warn);

    @Positive
        public SearchPath addFiles(Iterable<? extends Path> files);

    @Positive
        public void addFile(Path file, boolean warn);
    @Positive
    }

    @Positive
    protected static abstract class LocationHandler {

    @Positive
        abstract boolean handleOption(Option option, String value);

    @Positive
        boolean isSet();

    @Positive
        abstract boolean isExplicit();

    @Positive
        abstract Collection<Path> getPaths();

    @Positive
        abstract void setPaths(Iterable<? extends Path> paths) throws IOException;

    @Positive
        abstract void setPathsForModule(String moduleName, Iterable<? extends Path> paths) throws IOException;

    @Positive
        Location getLocationForModule(String moduleName) throws IOException;

    @Positive
        Location getLocationForModule(Path file) throws IOException;

    @Positive
        String inferModuleName();

    @Positive
        Iterable<Set<Location>> listLocationsForModules() throws IOException;

    @Positive
        @Pure
    @Positive
        abstract boolean contains(Path file) throws IOException;
    @Positive
    }

    @Positive
    private static abstract class BasicLocationHandler extends LocationHandler {

    @Positive
        protected BasicLocationHandler(Location location, Option... options) {
    @Positive
        }

    @Positive
        @Override
    @Positive
        void setPathsForModule(String moduleName, Iterable<? extends Path> files) throws IOException;

    @Positive
        protected Path checkSingletonDirectory(Iterable<? extends Path> paths) throws IOException;

    @Positive
        protected Path checkDirectory(Path path) throws IOException;

    @Positive
        @Override
    @Positive
        boolean isExplicit();
    @Positive
    }

    @Positive
    private class OutputLocationHandler extends BasicLocationHandler {

    @Positive
        @Override
    @Positive
        boolean handleOption(Option option, String value);

    @Positive
        @Override
    @Positive
        Collection<Path> getPaths();

    @Positive
        @Override
    @Positive
        void setPaths(Iterable<? extends Path> paths) throws IOException;

    @Positive
        @Override
    @Positive
        Location getLocationForModule(String name);

    @Positive
        @Override
    @Positive
        void setPathsForModule(String name, Iterable<? extends Path> paths) throws IOException;

    @Positive
        @Override
    @Positive
        Location getLocationForModule(Path file);

    @Positive
        @Override
    @Positive
        Iterable<Set<Location>> listLocationsForModules() throws IOException;

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        boolean contains(Path file) throws IOException;
    @Positive
    }

    @Positive
    private class SimpleLocationHandler extends BasicLocationHandler {

    @Positive
        protected Collection<Path> searchPath;

    @Positive
        @Override
    @Positive
        boolean handleOption(Option option, String value);

    @Positive
        @Override
    @Positive
        Collection<Path> getPaths();

    @Positive
        @Override
    @Positive
        void setPaths(Iterable<? extends Path> files);

    @Positive
        protected SearchPath computePath(String value);

    @Positive
        protected SearchPath createPath();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        boolean contains(Path file) throws IOException;
    @Positive
    }

    @Positive
    private class ClassPathLocationHandler extends SimpleLocationHandler {

    @Positive
        @Override
    @Positive
        Collection<Path> getPaths();

    @Positive
        @Override
    @Positive
        protected SearchPath computePath(String value);

    @Positive
        @Override
    @Positive
        protected SearchPath createPath();
    @Positive
    }

    @Positive
    private class BootClassPathLocationHandler extends BasicLocationHandler {

    @Positive
        boolean isDefault();

    @Positive
        @Override
    @Positive
        boolean handleOption(Option option, String value);

    @Positive
        @Override
    @Positive
        Collection<Path> getPaths();

    @Positive
        @Override
    @Positive
        void setPaths(Iterable<? extends Path> files);

    @Positive
        SearchPath computePath() throws IOException;

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        boolean contains(Path file) throws IOException;
    @Positive
    }

    @Positive
    private class ModuleLocationHandler extends LocationHandler implements Location {

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public String getName();

    @Positive
        @Override
    @Positive
        @DefinedBy(Api.COMPILER)
    @Positive
        public boolean isOutputLocation();

    @Positive
        @Override
    @Positive
        boolean handleOption(Option option, String value);

    @Positive
        @Override
    @Positive
        Collection<Path> getPaths();

    @Positive
        @Override
    @Positive
        boolean isExplicit();

    @Positive
        @Override
    @Positive
        void setPaths(Iterable<? extends Path> paths) throws IOException;

    @Positive
        @Override
    @Positive
        void setPathsForModule(String moduleName, Iterable<? extends Path> paths);

    @Positive
        @Override
    @Positive
        String inferModuleName();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        boolean contains(Path file) throws IOException;

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    private class ModuleTable {

    @Positive
        void add(ModuleLocationHandler h);

    @Positive
        void updatePaths(ModuleLocationHandler h);

    @Positive
        ModuleLocationHandler get(String name);

    @Positive
        ModuleLocationHandler get(Path path);

    @Positive
        void clear();

    @Positive
        boolean isEmpty();

    @Positive
        @Pure
    @Positive
        boolean contains(Path file) throws IOException;

    @Positive
        Set<Location> locations();

    @Positive
        Set<Location> explicitLocations();
    @Positive
    }

    @Positive
    private class ModulePathLocationHandler extends SimpleLocationHandler {

    @Positive
        @Override
    @Positive
        public boolean handleOption(Option option, String value);

    @Positive
        @Override
    @Positive
        public Location getLocationForModule(String moduleName);

    @Positive
        @Override
    @Positive
        public Location getLocationForModule(Path file);

    @Positive
        @Override
    @Positive
        Iterable<Set<Location>> listLocationsForModules();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        boolean contains(Path file) throws IOException;

    @Positive
        @Override
    @Positive
        void setPaths(Iterable<? extends Path> paths);

    @Positive
        @Override
    @Positive
        void setPathsForModule(String name, Iterable<? extends Path> paths) throws IOException;

    @Positive
        class ModulePathIterator implements Iterator<Set<Location>> {

    @Positive
            @Override
    @Positive
            public boolean hasNext();

    @Positive
            @Override
    @Positive
            public Set<Location> next();
    @Positive
        }
    @Positive
    }

    @Positive
    private class ModuleSourcePathLocationHandler extends BasicLocationHandler {

    @Positive
        @Override
    @Positive
        boolean handleOption(Option option, String value);

    @Positive
        void init(String value);

    @Positive
        void initForModule(String value);

    @Positive
        void initFromPattern(String value);

    @Positive
        void add(Map<String, List<Path>> map, Path prefix, Path suffix);

    @Positive
        int getMatchingBrace(String value, int offset);

    @Positive
        @Override
    @Positive
        boolean isSet();

    @Positive
        @Override
    @Positive
        Collection<Path> getPaths();

    @Positive
        @Override
    @Positive
        void setPaths(Iterable<? extends Path> files) throws IOException;

    @Positive
        @Override
    @Positive
        void setPathsForModule(String name, Iterable<? extends Path> paths) throws IOException;

    @Positive
        @Override
    @Positive
        Location getLocationForModule(String name);

    @Positive
        @Override
    @Positive
        Location getLocationForModule(Path file);

    @Positive
        @Override
    @Positive
        Iterable<Set<Location>> listLocationsForModules();

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        boolean contains(Path file) throws IOException;
    @Positive
    }

    @Positive
    private class SystemModulesLocationHandler extends BasicLocationHandler {

    @Positive
        @Override
    @Positive
        boolean handleOption(Option option, String value);

    @Positive
        @Override
    @Positive
        Collection<Path> getPaths();

    @Positive
        @Override
    @Positive
        void setPaths(Iterable<? extends Path> files) throws IOException;

    @Positive
        @Override
    @Positive
        void setPathsForModule(String name, Iterable<? extends Path> paths) throws IOException;

    @Positive
        @Override
    @Positive
        Location getLocationForModule(String name) throws IOException;

    @Positive
        @Override
    @Positive
        Location getLocationForModule(Path file) throws IOException;

    @Positive
        @Override
    @Positive
        Iterable<Set<Location>> listLocationsForModules() throws IOException;

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        boolean contains(Path file) throws IOException;
    @Positive
    }

    @Positive
    private class PatchModulesLocationHandler extends BasicLocationHandler {

    @Positive
        @Override
    @Positive
        boolean handleOption(Option option, String value);

    @Positive
        @Override
    @Positive
        boolean isSet();

    @Positive
        @Override
    @Positive
        Collection<Path> getPaths();

    @Positive
        @Override
    @Positive
        void setPaths(Iterable<? extends Path> files) throws IOException;

    @Positive
        @Override
    @Positive
        void setPathsForModule(String moduleName, Iterable<? extends Path> files) throws IOException;

    @Positive
        @Override
    @Positive
        Location getLocationForModule(String name) throws IOException;

    @Positive
        @Override
    @Positive
        Location getLocationForModule(Path file) throws IOException;

    @Positive
        @Override
    @Positive
        Iterable<Set<Location>> listLocationsForModules() throws IOException;

    @Positive
        @Override
    @Positive
        @Pure
    @Positive
        boolean contains(Path file) throws IOException;
    @Positive
    }

    @Positive
    void initHandlers();

    @Positive
    boolean handleOption(Option option, String value);

    @Positive
    boolean hasLocation(Location location);

    @Positive
    boolean hasExplicitLocation(Location location);

    @Positive
    Collection<Path> getLocation(Location location);

    @Positive
    Path getOutputLocation(Location location);

    @Positive
    void setLocation(Location location, Iterable<? extends Path> files) throws IOException;

    @Positive
    Location getLocationForModule(Location location, String name) throws IOException;

    @Positive
    Location getLocationForModule(Location location, Path file) throws IOException;

    @Positive
    void setLocationForModule(Location location, String moduleName, Iterable<? extends Path> files) throws IOException;

    @Positive
    String inferModuleName(Location location);

    @Positive
    Iterable<Set<Location>> listLocationsForModules(Location location) throws IOException;

    @Positive
    @Pure
    @Positive
    boolean contains(Location location, Path file) throws IOException;

    @Positive
    protected LocationHandler getHandler(Location location);

    @Positive
    static Path normalize(Path p);
    @Positive
}

// CFWR semantic augmentation - variant 1
