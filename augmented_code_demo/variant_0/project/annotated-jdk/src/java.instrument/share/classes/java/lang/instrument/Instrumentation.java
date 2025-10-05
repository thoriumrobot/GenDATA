/*
    @Positive
 * Copyright (c) 2003, 2019, Oracle and/or its affiliates. All rights reserved.
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
package java.lang.instrument;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.security.ProtectionDomain;
    @Positive
import java.util.List;
    @Positive
import java.util.Map;
    @Positive
import java.util.Set;
    @Positive
import java.util.jar.JarFile;

    @Positive
@AnnotatedFor({ "nullness" })
    @Positive
public interface Instrumentation {

    @Positive
    void addTransformer(ClassFileTransformer transformer, boolean canRetransform);

    @Positive
    void addTransformer(ClassFileTransformer transformer);

    @Positive
    boolean removeTransformer(ClassFileTransformer transformer);

    @Positive
    boolean isRetransformClassesSupported();

    @Positive
    void retransformClasses(Class<?>... classes) throws UnmodifiableClassException;

    @Positive
    boolean isRedefineClassesSupported();

    @Positive
    void redefineClasses(ClassDefinition... definitions) throws ClassNotFoundException, UnmodifiableClassException;

    @Positive
    boolean isModifiableClass(Class<?> theClass);

    @Positive
    @SuppressWarnings("rawtypes")
    @Positive
    Class[] getAllLoadedClasses();

    @Positive
    @SuppressWarnings("rawtypes")
    @Positive
    Class[] getInitiatedClasses(@Nullable ClassLoader loader);

    @Positive
    long getObjectSize(@Nullable Object objectToSize);

    @Positive
    void appendToBootstrapClassLoaderSearch(JarFile jarfile);

    @Positive
    void appendToSystemClassLoaderSearch(JarFile jarfile);

    @Positive
    boolean isNativeMethodPrefixSupported();

    @Positive
    void setNativeMethodPrefix(ClassFileTransformer transformer, @Nullable String prefix);

    @Positive
    void redefineModule(Module module, Set<Module> extraReads, Map<String, Set<Module>> extraExports, Map<String, Set<Module>> extraOpens, Set<Class<?>> extraUses, Map<Class<?>, List<Class<?>>> extraProvides);

    @Positive
    boolean isModifiableModule(Module module);
    @Positive
}

// CFWR semantic augmentation - variant 0
