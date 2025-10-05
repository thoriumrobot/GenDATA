/*
 * Copyright (c) 2003, 2021, Oracle and/or its affiliates. All rights reserved.
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
package com.sun.tools.javac.file;

import org.checkerframework.dataflow.qual.Pure;
import java.io.Closeable;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.DirectoryIteratorException;
import java.nio.file.DirectoryStream;
import java.nio.file.FileSystem;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.ProviderNotFoundException;
import java.nio.file.spi.FileSystemProvider;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.NoSuchElementException;
import java.util.Set;
import java.util.function.Predicate;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.jar.Attributes;
import java.util.jar.Manifest;
import javax.lang.model.SourceVersion;
import javax.tools.JavaFileManager;
import javax.tools.JavaFileManager.Location;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.StandardJavaFileManager.PathFactory;
import javax.tools.StandardLocation;
import jdk.internal.jmod.JmodFile;
import com.sun.tools.javac.code.Lint;
import com.sun.tools.javac.code.Lint.LintCategory;
import com.sun.tools.javac.main.Option;
import com.sun.tools.javac.resources.CompilerProperties.Errors;
import com.sun.tools.javac.resources.CompilerProperties.Warnings;
import com.sun.tools.javac.util.DefinedBy;
import com.sun.tools.javac.util.DefinedBy.Api;
import com.sun.tools.javac.util.JCDiagnostic.Warning;
import com.sun.tools.javac.util.ListBuffer;
import com.sun.tools.javac.util.Log;
import com.sun.tools.javac.jvm.ModuleNameReader;
import com.sun.tools.javac.util.Iterators;
import com.sun.tools.javac.util.Pair;
import com.sun.tools.javac.util.StringUtils;
import static javax.tools.StandardLocation.SYSTEM_MODULES;
import static javax.tools.StandardLocation.PLATFORM_CLASS_PATH;
import static com.sun.tools.javac.main.Option.BOOT_CLASS_PATH;
import static com.sun.tools.javac.main.Option.ENDORSEDDIRS;
import static com.sun.tools.javac.main.Option.EXTDIRS;
import static com.sun.tools.javac.main.Option.XBOOTCLASSPATH_APPEND;
import static com.sun.tools.javac.main.Option.XBOOTCLASSPATH_PREPEND;

public class Locations {

    Path getPath(String first, String... more);

    public void close() throws IOException;

    void update(Log log, boolean warn, FSInfo fsInfo);

    void setPathFactory(PathFactory f);

    boolean isDefaultBootClassPath();

    boolean isDefaultSystemModulesPath();

    public void setMultiReleaseValue(String multiReleaseValue);

    private class SearchPath extends LinkedHashSet<Path> {

        public SearchPath expandJarClassPaths(boolean x);

        public SearchPath emptyPathDefault(Path x);

        public SearchPath addDirectories(String dirs, boolean warn);

        public SearchPath addDirectories(String dirs);

        public SearchPath addFiles(String files, boolean warn);

        public SearchPath addFiles(String files);

        public SearchPath addFiles(Iterable<? extends Path> files, boolean warn);

        public SearchPath addFiles(Iterable<? extends Path> files);

        public void addFile(Path file, boolean warn);
    }

    protected static abstract class LocationHandler {

        abstract boolean handleOption(Option option, String value);

        boolean isSet();

        abstract boolean isExplicit();

        abstract Collection<Path> getPaths();

        abstract void setPaths(Iterable<? extends Path> paths) throws IOException;

        abstract void setPathsForModule(String moduleName, Iterable<? extends Path> paths) throws IOException;

        Location getLocationForModule(String moduleName) throws IOException;

        Location getLocationForModule(Path file) throws IOException;

        String inferModuleName();

        Iterable<Set<Location>> listLocationsForModules() throws IOException;

        @Pure
        abstract boolean contains(Path file) throws IOException;
    }

    private static abstract class BasicLocationHandler extends LocationHandler {

        protected BasicLocationHandler(Location location, Option... options) {
        }

        @Override
        void setPathsForModule(String moduleName, Iterable<? extends Path> files) throws IOException;

        protected Path checkSingletonDirectory(Iterable<? extends Path> paths) throws IOException;

        protected Path checkDirectory(Path path) throws IOException;

        @Override
        boolean isExplicit();
    }

    private class OutputLocationHandler extends BasicLocationHandler {

        @Override
        boolean handleOption(Option option, String value);

        @Override
        Collection<Path> getPaths();

        @Override
        void setPaths(Iterable<? extends Path> paths) throws IOException;

        @Override
        Location getLocationForModule(String name);

        @Override
        void setPathsForModule(String name, Iterable<? extends Path> paths) throws IOException;

        @Override
        Location getLocationForModule(Path file);

        @Override
        Iterable<Set<Location>> listLocationsForModules() throws IOException;

        @Override
        @Pure
        boolean contains(Path file) throws IOException;
    }

    private class SimpleLocationHandler extends BasicLocationHandler {

        protected Collection<Path> searchPath;

        @Override
        boolean handleOption(Option option, String value);

        @Override
        Collection<Path> getPaths();

        @Override
        void setPaths(Iterable<? extends Path> files);

        protected SearchPath computePath(String value);

        protected SearchPath createPath();

        @Override
        @Pure
        boolean contains(Path file) throws IOException;
    }

    private class ClassPathLocationHandler extends SimpleLocationHandler {

        @Override
        Collection<Path> getPaths();

        @Override
        protected SearchPath computePath(String value);

        @Override
        protected SearchPath createPath();
    }

    private class BootClassPathLocationHandler extends BasicLocationHandler {

        boolean isDefault();

        @Override
        boolean handleOption(Option option, String value);

        @Override
        Collection<Path> getPaths();

        @Override
        void setPaths(Iterable<? extends Path> files);

        SearchPath computePath() throws IOException;

        @Override
        @Pure
        boolean contains(Path file) throws IOException;
    }

    private class ModuleLocationHandler extends LocationHandler implements Location {

        @Override
        @DefinedBy(Api.COMPILER)
        public String getName();

        @Override
        @DefinedBy(Api.COMPILER)
        public boolean isOutputLocation();

        @Override
        boolean handleOption(Option option, String value);

        @Override
        Collection<Path> getPaths();

        @Override
        boolean isExplicit();

        @Override
        void setPaths(Iterable<? extends Path> paths) throws IOException;

        @Override
        void setPathsForModule(String moduleName, Iterable<? extends Path> paths);

        @Override
        String inferModuleName();

        @Override
        @Pure
        boolean contains(Path file) throws IOException;

        @Override
        public String toString();
    }

    private class ModuleTable {

        void add(ModuleLocationHandler h);

        void updatePaths(ModuleLocationHandler h);

        ModuleLocationHandler get(String name);

        ModuleLocationHandler get(Path path);

        void clear();

        boolean isEmpty();

        @Pure
        boolean contains(Path file) throws IOException;

        Set<Location> locations();

        Set<Location> explicitLocations();
    }

    private class ModulePathLocationHandler extends SimpleLocationHandler {

        @Override
        public boolean handleOption(Option option, String value);

        @Override
        public Location getLocationForModule(String moduleName);

        @Override
        public Location getLocationForModule(Path file);

        @Override
        Iterable<Set<Location>> listLocationsForModules();

        @Override
        @Pure
        boolean contains(Path file) throws IOException;

        @Override
        void setPaths(Iterable<? extends Path> paths);

        @Override
        void setPathsForModule(String name, Iterable<? extends Path> paths) throws IOException;

        class ModulePathIterator implements Iterator<Set<Location>> {

            @Override
            public boolean hasNext();

            @Override
            public Set<Location> next();
        }
    }

    private class ModuleSourcePathLocationHandler extends BasicLocationHandler {

        @Override
        boolean handleOption(Option option, String value);

        void init(String value);

        void initForModule(String value);

        void initFromPattern(String value);

        void add(Map<String, List<Path>> map, Path prefix, Path suffix);

        int getMatchingBrace(String value, int offset);

        @Override
        boolean isSet();

        @Override
        Collection<Path> getPaths();

        @Override
        void setPaths(Iterable<? extends Path> files) throws IOException;

        @Override
        void setPathsForModule(String name, Iterable<? extends Path> paths) throws IOException;

        @Override
        Location getLocationForModule(String name);

        @Override
        Location getLocationForModule(Path file);

        @Override
        Iterable<Set<Location>> listLocationsForModules();

        @Override
        @Pure
        boolean contains(Path file) throws IOException;
    }

    private class SystemModulesLocationHandler extends BasicLocationHandler {

        @Override
        boolean handleOption(Option option, String value);

        @Override
        Collection<Path> getPaths();

        @Override
        void setPaths(Iterable<? extends Path> files) throws IOException;

        @Override
        void setPathsForModule(String name, Iterable<? extends Path> paths) throws IOException;

        @Override
        Location getLocationForModule(String name) throws IOException;

        @Override
        Location getLocationForModule(Path file) throws IOException;

        @Override
        Iterable<Set<Location>> listLocationsForModules() throws IOException;

        @Override
        @Pure
        boolean contains(Path file) throws IOException;
    }

    private class PatchModulesLocationHandler extends BasicLocationHandler {

        @Override
        boolean handleOption(Option option, String value);

        @Override
        boolean isSet();

        @Override
        Collection<Path> getPaths();

        @Override
        void setPaths(Iterable<? extends Path> files) throws IOException;

        @Override
        void setPathsForModule(String moduleName, Iterable<? extends Path> files) throws IOException;

        @Override
        Location getLocationForModule(String name) throws IOException;

        @Override
        Location getLocationForModule(Path file) throws IOException;

        @Override
        Iterable<Set<Location>> listLocationsForModules() throws IOException;

        @Override
        @Pure
        boolean contains(Path file) throws IOException;
    }

    void initHandlers();

    boolean handleOption(Option option, String value);

    boolean hasLocation(Location location);

    boolean hasExplicitLocation(Location location);

    Collection<Path> getLocation(Location location);

    Path getOutputLocation(Location location);

    void setLocation(Location location, Iterable<? extends Path> files) throws IOException;

    Location getLocationForModule(Location location, String name) throws IOException;

    Location getLocationForModule(Location location, Path file) throws IOException;

    void setLocationForModule(Location location, String moduleName, Iterable<? extends Path> files) throws IOException;

    String inferModuleName(Location location);

    Iterable<Set<Location>> listLocationsForModules(Location location) throws IOException;

    @Pure
    boolean contains(Location location, Path file) throws IOException;

    protected LocationHandler getHandler(Location location);

    static Path normalize(Path p);
}
