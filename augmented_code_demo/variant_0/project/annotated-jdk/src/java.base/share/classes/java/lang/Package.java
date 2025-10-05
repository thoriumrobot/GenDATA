/*
    @Positive
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.signature.qual.DotSeparatedIdentifiers;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import java.lang.annotation.Annotation;
    @Positive
import java.lang.reflect.AnnotatedElement;
    @Positive
import java.net.MalformedURLException;
    @Positive
import java.net.URI;
    @Positive
import java.net.URL;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Objects;
    @Positive
import jdk.internal.loader.BootLoader;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;

    @Positive
@AnnotatedFor({ "interning", "lock", "nullness", "signature" })
    @Positive
@UsesObjectEquals
    @Positive
public class Package extends NamedPackage implements java.lang.reflect.AnnotatedElement {

    @Positive
    @DotSeparatedIdentifiers
    @Positive
    public String getName();

    @Positive
    @Nullable
    @Positive
    public String getSpecificationTitle();

    @Positive
    @Nullable
    @Positive
    public String getSpecificationVersion();

    @Positive
    @Nullable
    @Positive
    public String getSpecificationVendor();

    @Positive
    @Nullable
    @Positive
    public String getImplementationTitle();

    @Positive
    @Nullable
    @Positive
    public String getImplementationVersion();

    @Positive
    @Nullable
    @Positive
    public String getImplementationVendor();

    @Positive
    @Pure
    @Positive
    public boolean isSealed(@GuardSatisfied Package this);

    @Positive
    @Pure
    @Positive
    public boolean isSealed(@GuardSatisfied Package this, @GuardSatisfied URL url);

    @Positive
    @Pure
    @Positive
    public boolean isCompatibleWith(@GuardSatisfied Package this, String desired) throws NumberFormatException;

    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    @Deprecated()
    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    @Nullable
    @Positive
    public static Package getPackage(@DotSeparatedIdentifiers String name);

    @Positive
    @Pure
    @Positive
    @CallerSensitive
    @Positive
    public static Package[] getPackages();

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public int hashCode(@GuardSatisfied Package this);

    @Positive
    @SideEffectFree
    @Positive
    @Override
    @Positive
    public String toString(@GuardSatisfied Package this);

    @Positive
    @Override
    @Positive
    @Nullable
    @Positive
    public <A extends Annotation> A getAnnotation(Class<A> annotationClass);

    @Positive
    @Pure
    @Positive
    @Override
    @Positive
    public boolean isAnnotationPresent(@GuardSatisfied Package this, @GuardSatisfied Class<? extends Annotation> annotationClass);

    @Positive
    @Override
    @Positive
    public <A extends Annotation> A[] getAnnotationsByType(Class<A> annotationClass);

    @Positive
    @Override
    @Positive
    public Annotation[] getAnnotations();

    @Positive
    @Override
    @Positive
    @Nullable
    @Positive
    public <A extends Annotation> A getDeclaredAnnotation(Class<A> annotationClass);

    @Positive
    @Override
    @Positive
    public <A extends Annotation> A[] getDeclaredAnnotationsByType(Class<A> annotationClass);

    @Positive
    @Override
    @Positive
    public Annotation[] getDeclaredAnnotations();

    @Positive
    static class VersionInfo {

    @Positive
        static VersionInfo getInstance(String spectitle, String specversion, String specvendor, String impltitle, String implversion, String implvendor, URL sealbase);
    @Positive
    }
    @Positive
}

// CFWR semantic augmentation - variant 0
