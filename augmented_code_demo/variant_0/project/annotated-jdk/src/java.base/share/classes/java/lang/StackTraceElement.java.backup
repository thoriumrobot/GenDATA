/*
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signature.qual.FullyQualifiedName;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import jdk.internal.loader.BuiltinClassLoader;
import jdk.internal.misc.VM;
import jdk.internal.module.ModuleHashes;
import jdk.internal.module.ModuleReferenceImpl;
import java.lang.module.ModuleDescriptor.Version;
import java.lang.module.ModuleReference;
import java.lang.module.ResolvedModule;
import java.util.HashSet;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

@AnnotatedFor({ "lock", "nullness", "signature" })
public final class StackTraceElement implements java.io.Serializable {

    public StackTraceElement(@FullyQualifiedName String declaringClass, String methodName, @Nullable String fileName, int lineNumber) {
    }

    public StackTraceElement(String classLoaderName, String moduleName, String moduleVersion, String declaringClass, String methodName, String fileName, int lineNumber) {
    }

    @Nullable
    public String getFileName(@GuardSatisfied StackTraceElement this);

    public int getLineNumber(@GuardSatisfied StackTraceElement this);

    public String getModuleName();

    public String getModuleVersion();

    public String getClassLoaderName();

    @FullyQualifiedName
    public String getClassName(@GuardSatisfied StackTraceElement this);

    public String getMethodName(@GuardSatisfied StackTraceElement this);

    @Pure
    public boolean isNativeMethod(@GuardSatisfied StackTraceElement this);

    @SideEffectFree
    public String toString(@GuardSatisfied StackTraceElement this);

    @Pure
    public boolean equals(@GuardSatisfied StackTraceElement this, @GuardSatisfied @Nullable Object obj);

    @Pure
    public int hashCode(@GuardSatisfied StackTraceElement this);

    private static class HashedModules {

        static Set<String> hashedModules();

        @Pure
        static boolean contains(Module m);
    }

    static StackTraceElement[] of(Throwable x, int depth);

    static StackTraceElement of(StackFrameInfo sfi);
}
