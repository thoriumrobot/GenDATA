/*
    @Positive
 * Copyright (c) 2005, 2020, Oracle and/or its affiliates. All rights reserved.
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
package javax.tools;

    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import java.io.IOException;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Objects;
    @Positive
import java.util.ServiceLoader;
    @Positive
import java.util.Set;
    @Positive
import javax.tools.JavaFileObject.Kind;

    @Positive
public class ForwardingJavaFileManager<M extends JavaFileManager> implements JavaFileManager {

    @Positive
    protected final M fileManager;

    @Positive
    protected ForwardingJavaFileManager(M fileManager) {
    @Positive
    }

    @Positive
    @Override
    @Positive
    public ClassLoader getClassLoader(Location location);

    @Positive
    @Override
    @Positive
    public Iterable<JavaFileObject> list(Location location, String packageName, Set<Kind> kinds, boolean recurse) throws IOException;

    @Positive
    @Override
    @Positive
    public String inferBinaryName(Location location, JavaFileObject file);

    @Positive
    @Override
    @Positive
    public boolean isSameFile(FileObject a, FileObject b);

    @Positive
    @Override
    @Positive
    public boolean handleOption(String current, Iterator<String> remaining);

    @Positive
    @Override
    @Positive
    public boolean hasLocation(Location location);

    @Positive
    @Override
    @Positive
    public int isSupportedOption(String option);

    @Positive
    @Override
    @Positive
    public JavaFileObject getJavaFileForInput(Location location, String className, Kind kind) throws IOException;

    @Positive
    @Override
    @Positive
    public JavaFileObject getJavaFileForOutput(Location location, String className, Kind kind, FileObject sibling) throws IOException;

    @Positive
    @Override
    @Positive
    public FileObject getFileForInput(Location location, String packageName, String relativeName) throws IOException;

    @Positive
    @Override
    @Positive
    public FileObject getFileForOutput(Location location, String packageName, String relativeName, FileObject sibling) throws IOException;

    @Positive
    @Override
    @Positive
    public void flush() throws IOException;

    @Positive
    @Override
    @Positive
    public void close() throws IOException;

    @Positive
    @Override
    @Positive
    public Location getLocationForModule(Location location, String moduleName) throws IOException;

    @Positive
    @Override
    @Positive
    public Location getLocationForModule(Location location, JavaFileObject fo) throws IOException;

    @Positive
    @Override
    @Positive
    public <S> ServiceLoader<S> getServiceLoader(Location location, Class<S> service) throws IOException;

    @Positive
    @Override
    @Positive
    public String inferModuleName(Location location) throws IOException;

    @Positive
    @Override
    @Positive
    public Iterable<Set<Location>> listLocationsForModules(Location location) throws IOException;

    @Positive
    @Override
    @Positive
    @Pure
    @Positive
    public boolean contains(Location location, FileObject fo) throws IOException;
    @Positive
}

// CFWR semantic augmentation - variant 0
