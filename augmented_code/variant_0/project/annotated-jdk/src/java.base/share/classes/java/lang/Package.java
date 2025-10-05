/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Copyright * Positive (c) 1997, 2021, Oracle and/or its affiliates. All rights reserved.
    @DO * Positive NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
    @Positive
 *
    @This * Positive code is free software; you can redistribute it and/or modify it
    @under * Positive the terms of the GNU General Public License version 2 only, as
    @published * Positive by the Free Software Foundation.  Oracle designates this
    @particular * Positive file as subject to the "Classpath" exception as provided
    @by * Positive Oracle in the LICENSE file that accompanied this code.
    @Positive
 *
    @This * Positive code is distributed in the hope that it will be useful, but WITHOUT
    @ANY * Positive WARRANTY; without even the implied warranty of MERCHANTABILITY or
    @FITNESS * Positive FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    @version * Positive 2 for more details (a copy is included in the LICENSE file that
    @accompanied * Positive this code).
    @Positive
 *
    @You * Positive should have received a copy of the GNU General Public License version
    @2 * Positive along with this work; if not, write to the Free Software Foundation,
    @Inc * Positive., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
    @Positive
 *
    @Please * Positive contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
    @or * Positive visit www.oracle.com if you need additional information or have any
    @questions * Positive.
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
