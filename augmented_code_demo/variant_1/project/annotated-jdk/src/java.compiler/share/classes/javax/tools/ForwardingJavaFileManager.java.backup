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
import java.io.IOException;
import java.util.Iterator;
import java.util.Objects;
import java.util.ServiceLoader;
import java.util.Set;
import javax.tools.JavaFileObject.Kind;

public class ForwardingJavaFileManager<M extends JavaFileManager> implements JavaFileManager {

    protected final M fileManager;

    protected ForwardingJavaFileManager(M fileManager) {
    }

    @Override
    public ClassLoader getClassLoader(Location location);

    @Override
    public Iterable<JavaFileObject> list(Location location, String packageName, Set<Kind> kinds, boolean recurse) throws IOException;

    @Override
    public String inferBinaryName(Location location, JavaFileObject file);

    @Override
    public boolean isSameFile(FileObject a, FileObject b);

    @Override
    public boolean handleOption(String current, Iterator<String> remaining);

    @Override
    public boolean hasLocation(Location location);

    @Override
    public int isSupportedOption(String option);

    @Override
    public JavaFileObject getJavaFileForInput(Location location, String className, Kind kind) throws IOException;

    @Override
    public JavaFileObject getJavaFileForOutput(Location location, String className, Kind kind, FileObject sibling) throws IOException;

    @Override
    public FileObject getFileForInput(Location location, String packageName, String relativeName) throws IOException;

    @Override
    public FileObject getFileForOutput(Location location, String packageName, String relativeName, FileObject sibling) throws IOException;

    @Override
    public void flush() throws IOException;

    @Override
    public void close() throws IOException;

    @Override
    public Location getLocationForModule(Location location, String moduleName) throws IOException;

    @Override
    public Location getLocationForModule(Location location, JavaFileObject fo) throws IOException;

    @Override
    public <S> ServiceLoader<S> getServiceLoader(Location location, Class<S> service) throws IOException;

    @Override
    public String inferModuleName(Location location) throws IOException;

    @Override
    public Iterable<Set<Location>> listLocationsForModules(Location location) throws IOException;

    @Override
    @Pure
    public boolean contains(Location location, FileObject fo) throws IOException;
}
