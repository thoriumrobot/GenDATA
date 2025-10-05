/*
    @Positive
 * Copyright (c) 2000, 2020, Oracle and/or its affiliates. All rights reserved.
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
package java.lang;

    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signature.qual.FullyQualifiedName;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import jdk.internal.loader.BuiltinClassLoader;
    @Positive
import jdk.internal.misc.VM;
    @Positive
import jdk.internal.module.ModuleHashes;
    @Positive
import jdk.internal.module.ModuleReferenceImpl;
    @Positive
import java.lang.module.ModuleDescriptor.Version;
    @Positive
import java.lang.module.ModuleReference;
    @Positive
import java.lang.module.ResolvedModule;
    @Positive
import java.util.HashSet;
    @Positive
import java.util.Objects;
    @Positive
import java.util.Optional;
    @Positive
import java.util.Set;

    @Positive
@AnnotatedFor({ "lock", "nullness", "signature" })
    @Positive
public final class StackTraceElement implements java.io.Serializable {

    @Positive
    public StackTraceElement(@FullyQualifiedName String declaringClass, String methodName, @Nullable String fileName, int lineNumber) {
    @Positive
    }

    @Positive
    public StackTraceElement(String classLoaderName, String moduleName, String moduleVersion, String declaringClass, String methodName, String fileName, int lineNumber) {
    @Positive
    }

    @Positive
    @Nullable
    @Positive
    public String getFileName(@GuardSatisfied StackTraceElement this);

    @Positive
    public int getLineNumber(@GuardSatisfied StackTraceElement this);

    @Positive
    public String getModuleName();

    @Positive
    public String getModuleVersion();

    @Positive
    public String getClassLoaderName();

    @Positive
    @FullyQualifiedName
    @Positive
    public String getClassName(@GuardSatisfied StackTraceElement this);

    @Positive
    public String getMethodName(@GuardSatisfied StackTraceElement this);

    @Positive
    @Pure
    @Positive
    public boolean isNativeMethod(@GuardSatisfied StackTraceElement this);

    @Positive
    @SideEffectFree
    @Positive
    public String toString(@GuardSatisfied StackTraceElement this);

    @Positive
    @Pure
    @Positive
    public boolean equals(@GuardSatisfied StackTraceElement this, @GuardSatisfied @Nullable Object obj);

    @Positive
    @Pure
    @Positive
    public int hashCode(@GuardSatisfied StackTraceElement this);

    @Positive
    private static class HashedModules {

    @Positive
        static Set<String> hashedModules();

    @Positive
        @Pure
    @Positive
        static boolean contains(Module m);
    @Positive
    }

    @Positive
    static StackTraceElement[] of(Throwable x, int depth);

    @Positive
    static StackTraceElement of(StackFrameInfo sfi);
    @Positive
}

// CFWR semantic augmentation - variant 1
