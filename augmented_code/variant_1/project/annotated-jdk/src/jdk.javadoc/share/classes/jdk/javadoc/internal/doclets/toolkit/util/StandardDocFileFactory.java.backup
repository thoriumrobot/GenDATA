/*
 * Copyright (c) 1998, 2020, Oracle and/or its affiliates. All rights reserved.
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
package jdk.javadoc.internal.doclets.toolkit.util;

import org.checkerframework.dataflow.qual.Pure;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import javax.tools.DocumentationTool;
import javax.tools.FileObject;
import javax.tools.JavaFileManager.Location;
import javax.tools.JavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.StandardLocation;
import com.sun.tools.javac.util.Assert;
import jdk.javadoc.internal.doclets.toolkit.BaseConfiguration;

class StandardDocFileFactory extends DocFileFactory {

    public StandardDocFileFactory(BaseConfiguration configuration) {
    }

    @Override
    public void setDestDir(String destDirName) throws SimpleDocletException;

    @Override
    public DocFile createFileForDirectory(String file);

    @Override
    public DocFile createFileForInput(String file);

    @Override
    public DocFile createFileForInput(Path file);

    @Override
    public DocFile createFileForOutput(DocPath path);

    @Override
    Iterable<DocFile> list(Location location, DocPath path);

    class StandardDocFile extends DocFile {

        @Override
        public FileObject getFileObject();

        @Override
        public InputStream openInputStream() throws DocFileIOException;

        @Override
        public OutputStream openOutputStream() throws DocFileIOException;

        @Override
        public Writer openWriter() throws DocFileIOException, UnsupportedEncodingException;

        @Override
        public boolean canRead();

        @Override
        public boolean canWrite();

        @Override
        public boolean exists();

        @Override
        public String getName();

        @Override
        public String getPath();

        @Override
        @Pure
        public boolean isAbsolute();

        @Override
        @Pure
        public boolean isDirectory();

        @Override
        @Pure
        public boolean isFile();

        @Override
        @Pure
        public boolean isSameFile(DocFile other);

        @Override
        public Iterable<DocFile> list() throws DocFileIOException;

        @Override
        public boolean mkdirs();

        @Override
        public DocFile resolve(DocPath p);

        @Override
        public DocFile resolve(String p);

        @Override
        public DocFile resolveAgainst(Location locn);

        @Override
        public String toString();
    }
}
