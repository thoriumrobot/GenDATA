/*
    @Positive
 * Copyright (c) 2002, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.util.jar;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.io.IOException;
    @Positive
import java.net.URL;
    @Positive
import java.security.CodeSource;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.List;
    @Positive
import java.util.zip.ZipEntry;
    @Positive
import java.util.zip.ZipFile;
    @Positive
import jdk.internal.access.JavaUtilJarAccess;

    @Positive
@AnnotatedFor({ "interning" })
    @Positive
@UsesObjectEquals
    @Positive
class JavaUtilJarAccessImpl implements JavaUtilJarAccess {

    @Positive
    public boolean jarFileHasClassPathAttribute(JarFile jar) throws IOException;

    @Positive
    public CodeSource[] getCodeSources(JarFile jar, URL url);

    @Positive
    public CodeSource getCodeSource(JarFile jar, URL url, String name);

    @Positive
    public Enumeration<String> entryNames(JarFile jar, CodeSource[] cs);

    @Positive
    public Enumeration<JarEntry> entries2(JarFile jar);

    @Positive
    public void setEagerValidation(JarFile jar, boolean eager);

    @Positive
    public List<Object> getManifestDigests(JarFile jar);

    @Positive
    public Attributes getTrustedAttributes(Manifest man, String name);

    @Positive
    public void ensureInitialization(JarFile jar);

    @Positive
    public boolean isInitializing();

    @Positive
    public JarEntry entryFor(JarFile jar, String name);
    @Positive
}

// CFWR semantic augmentation - variant 1
