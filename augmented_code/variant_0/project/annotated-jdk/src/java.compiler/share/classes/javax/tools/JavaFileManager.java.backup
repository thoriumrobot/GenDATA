/*
 * Copyright (c) 2005, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.tools;

import org.checkerframework.dataflow.qual.Pure;
import java.io.Closeable;
import java.io.Flushable;
import java.io.IOException;
import java.util.Iterator;
import java.util.ServiceLoader;
import java.util.Set;
import static javax.tools.JavaFileObject.Kind;

public interface JavaFileManager extends Closeable, Flushable, OptionChecker {

    interface Location {

        String getName();

        boolean isOutputLocation();

        default boolean isModuleOrientedLocation();
    }

    ClassLoader getClassLoader(Location location);

    Iterable<JavaFileObject> list(Location location, String packageName, Set<Kind> kinds, boolean recurse) throws IOException;

    String inferBinaryName(Location location, JavaFileObject file);

    boolean isSameFile(FileObject a, FileObject b);

    boolean handleOption(String current, Iterator<String> remaining);

    boolean hasLocation(Location location);

    JavaFileObject getJavaFileForInput(Location location, String className, Kind kind) throws IOException;

    JavaFileObject getJavaFileForOutput(Location location, String className, Kind kind, FileObject sibling) throws IOException;

    FileObject getFileForInput(Location location, String packageName, String relativeName) throws IOException;

    FileObject getFileForOutput(Location location, String packageName, String relativeName, FileObject sibling) throws IOException;

    @Override
    void flush() throws IOException;

    @Override
    void close() throws IOException;

    default Location getLocationForModule(Location location, String moduleName) throws IOException;

    default Location getLocationForModule(Location location, JavaFileObject fo) throws IOException;

    default <S> ServiceLoader<S> getServiceLoader(Location location, Class<S> service) throws IOException;

    default String inferModuleName(Location location) throws IOException;

    default Iterable<Set<Location>> listLocationsForModules(Location location) throws IOException;

    @Pure
    default boolean contains(Location location, FileObject fo) throws IOException;
}
