/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 2015, 2021, Oracle and/or its affiliates. All rights reserved.
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
package sun.util.resources;

    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import java.lang.ref.ReferenceQueue;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.List;
    @Positive
import java.util.Locale;
    @Positive
import java.util.MissingResourceException;
    @Positive
import java.util.Objects;
    @Positive
import java.util.ResourceBundle;
    @Positive
import java.util.ServiceConfigurationError;
    @Positive
import java.util.ServiceLoader;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import java.util.spi.ResourceBundleProvider;
    @Positive
import jdk.internal.access.JavaUtilResourceBundleAccess;
    @Positive
import jdk.internal.access.SharedSecrets;

    @Positive
public abstract class Bundles {

    @Positive
    public static ResourceBundle of(String baseName, Locale locale, Strategy strategy);

    @Positive
    public static String toOtherBundleName(String baseName, String bundleName, Locale locale);

    @Positive
    public interface Strategy {

    @Positive
        List<Locale> getCandidateLocales(String baseName, Locale locale);

    @Positive
        String toBundleName(String baseName, Locale locale);

    @Positive
        Class<? extends ResourceBundleProvider> getResourceBundleProviderType(String baseName, Locale locale);
    @Positive
    }

    @Positive
    private static interface CacheKeyReference {

    @Positive
        CacheKey getCacheKey();
    @Positive
    }

    @Positive
    private static class BundleReference extends SoftReference<ResourceBundle> implements CacheKeyReference {

    @Positive
        @Override
    @Positive
        public CacheKey getCacheKey();
    @Positive
    }

    @Positive
    private static class CacheKey implements Cloneable {

    @Positive
        String getName();

    @Positive
        CacheKey setName(String baseName);

    @Positive
        Locale getLocale();

    @Positive
        CacheKey setLocale(Locale locale);

    @Positive
        ServiceLoader<ResourceBundleProvider> getProviders();

    @Positive
        void setProviders(ServiceLoader<ResourceBundleProvider> providers);

    @Positive
        @Override
    @Positive
        public boolean equals(Object other);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        @Override
    @Positive
        public Object clone();

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }
    @Positive
}
