/*
 * Copyright (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.signature.qual.DotSeparatedIdentifiers;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.lang.annotation.Annotation;
import java.lang.reflect.AnnotatedElement;
import java.net.MalformedURLException;
import java.net.URI;
import java.net.URL;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Objects;
import jdk.internal.loader.BootLoader;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.Reflection;

@AnnotatedFor({ "interning", "lock", "nullness", "signature" })
@UsesObjectEquals
public class Package extends NamedPackage implements java.lang.reflect.AnnotatedElement {

    @DotSeparatedIdentifiers
    public String getName();

    @Nullable
    public String getSpecificationTitle();

    @Nullable
    public String getSpecificationVersion();

    @Nullable
    public String getSpecificationVendor();

    @Nullable
    public String getImplementationTitle();

    @Nullable
    public String getImplementationVersion();

    @Nullable
    public String getImplementationVendor();

    @Pure
    public boolean isSealed(@GuardSatisfied Package this);

    @Pure
    public boolean isSealed(@GuardSatisfied Package this, @GuardSatisfied URL url);

    @Pure
    public boolean isCompatibleWith(@GuardSatisfied Package this, String desired) throws NumberFormatException;

    @Pure
    @CallerSensitive
    @Deprecated()
    @SuppressWarnings("deprecation")
    @Nullable
    public static Package getPackage(@DotSeparatedIdentifiers String name);

    @Pure
    @CallerSensitive
    public static Package[] getPackages();

    @Pure
    @Override
    public int hashCode(@GuardSatisfied Package this);

    @SideEffectFree
    @Override
    public String toString(@GuardSatisfied Package this);

    @Override
    @Nullable
    public <A extends Annotation> A getAnnotation(Class<A> annotationClass);

    @Pure
    @Override
    public boolean isAnnotationPresent(@GuardSatisfied Package this, @GuardSatisfied Class<? extends Annotation> annotationClass);

    @Override
    public <A extends Annotation> A[] getAnnotationsByType(Class<A> annotationClass);

    @Override
    public Annotation[] getAnnotations();

    @Override
    @Nullable
    public <A extends Annotation> A getDeclaredAnnotation(Class<A> annotationClass);

    @Override
    public <A extends Annotation> A[] getDeclaredAnnotationsByType(Class<A> annotationClass);

    @Override
    public Annotation[] getDeclaredAnnotations();

    static class VersionInfo {

        static VersionInfo getInstance(String spectitle, String specversion, String specvendor, String impltitle, String implversion, String implvendor, URL sealbase);
    }
}
