/*
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

import org.checkerframework.checker.i18n.qual.LocalizableKey;
import org.checkerframework.checker.i18n.qual.Localized;
import org.checkerframework.checker.i18nformatter.qual.I18nMakeFormat;
import org.checkerframework.checker.index.qual.NonNegative;
import org.checkerframework.checker.lock.qual.GuardSatisfied;
import org.checkerframework.checker.nullness.qual.EnsuresKeyForIf;
import org.checkerframework.checker.nullness.qual.KeyFor;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.propkey.qual.PropertyKey;
import org.checkerframework.checker.signature.qual.BinaryName;
import org.checkerframework.dataflow.qual.Pure;
import org.checkerframework.dataflow.qual.SideEffectFree;
import org.checkerframework.framework.qual.AnnotatedFor;
import org.checkerframework.framework.qual.CFComment;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.lang.ref.Reference;
import java.lang.ref.ReferenceQueue;
import java.lang.ref.SoftReference;
import java.lang.ref.WeakReference;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Modifier;
import java.net.JarURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.security.PrivilegedActionException;
import java.security.PrivilegedExceptionAction;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.jar.JarEntry;
import java.util.spi.ResourceBundleControlProvider;
import java.util.spi.ResourceBundleProvider;
import java.util.stream.Stream;
import jdk.internal.loader.BootLoader;
import jdk.internal.access.JavaUtilResourceBundleAccess;
import jdk.internal.access.SharedSecrets;
import jdk.internal.reflect.CallerSensitive;
import jdk.internal.reflect.Reflection;
import sun.security.action.GetPropertyAction;
import sun.util.locale.BaseLocale;
import sun.util.locale.LocaleObjectCache;
import sun.util.resources.Bundles;
import static sun.security.util.SecurityConstants.GET_CLASSLOADER_PERMISSION;

@AnnotatedFor({ "i18n", "i18nformatter", "index", "lock", "nullness", "propkey", "signature" })
public abstract class ResourceBundle {

    public String getBaseBundleName();

    protected ResourceBundle parent;

    public ResourceBundle() {
    }

    @I18nMakeFormat
    @Localized
    public final String getString(@LocalizableKey @PropertyKey String key);

    @Localized
    public final String[] getStringArray(@LocalizableKey @PropertyKey String key);

    @Localized
    public final Object getObject(@LocalizableKey @PropertyKey String key);

    public Locale getLocale();

    protected void setParent(ResourceBundle parent);

    private static final class CacheKey {

        String getName();

        Locale getLocale();

        CacheKey setLocale(Locale locale);

        Module getModule();

        Module getCallerModule();

        ServiceLoader<ResourceBundleProvider> getProviders();

        boolean hasProviders();

        boolean callerHasProvider();

        @Override
        public boolean equals(Object other);

        @Override
        public int hashCode();

        String getFormat();

        void setFormat(String format);

        @Override
        public String toString();
    }

    private static interface CacheKeyReference {

        public CacheKey getCacheKey();
    }

    private static class KeyElementReference<T> extends WeakReference<T> implements CacheKeyReference {

        @Override
        public CacheKey getCacheKey();
    }

    private static class BundleReference extends SoftReference<ResourceBundle> implements CacheKeyReference {

        @Override
        public CacheKey getCacheKey();
    }

    @CallerSensitive
    public static final ResourceBundle getBundle(@BinaryName String baseName);

    @CallerSensitive
    public static final ResourceBundle getBundle(@BinaryName String baseName, Control control);

    @CallerSensitive
    public static final ResourceBundle getBundle(@BinaryName String baseName, Locale locale);

    @CallerSensitive
    public static ResourceBundle getBundle(@BinaryName String baseName, Module module);

    @CallerSensitive
    public static ResourceBundle getBundle(@BinaryName String baseName, Locale targetLocale, Module module);

    @CallerSensitive
    public static final ResourceBundle getBundle(@BinaryName String baseName, Locale targetLocale, Control control);

    @CallerSensitive
    public static ResourceBundle getBundle(@BinaryName String baseName, Locale locale, ClassLoader loader);

    @CallerSensitive
    public static ResourceBundle getBundle(@BinaryName String baseName, Locale targetLocale, ClassLoader loader, Control control);

    private static class ResourceBundleControlProviderHolder {
    }

    @CallerSensitive
    public static final void clearCache();

    public static final void clearCache(ClassLoader loader);

    protected abstract Object handleGetObject(String key);

    @SideEffectFree
    public abstract Enumeration<String> getKeys(@GuardSatisfied ResourceBundle this);

    @Pure
    @EnsuresKeyForIf(result = true, expression = "#1", map = "this")
    public boolean containsKey(@GuardSatisfied ResourceBundle this, String key);

    @SideEffectFree
    public Set<@KeyFor("this") @LocalizableKey @PropertyKey String> keySet(@GuardSatisfied ResourceBundle this);

    protected Set<String> handleKeySet();

    public static class Control {

        public static final List<String> FORMAT_DEFAULT;

        public static final List<String> FORMAT_CLASS;

        public static final List<String> FORMAT_PROPERTIES;

        public static final long TTL_DONT_CACHE;

        public static final long TTL_NO_EXPIRATION_CONTROL;

        protected Control() {
        }

        public static final Control getControl(List<String> formats);

        public static final Control getNoFallbackControl(List<String> formats);

        public List<String> getFormats(String baseName);

        public List<Locale> getCandidateLocales(String baseName, Locale locale);

        private static class CandidateListCache extends LocaleObjectCache<BaseLocale, List<Locale>> {

            protected List<Locale> createObject(BaseLocale base);
        }

        public Locale getFallbackLocale(String baseName, Locale locale);

        public ResourceBundle newBundle(@BinaryName String baseName, Locale locale, String format, ClassLoader loader, boolean reload) throws IllegalAccessException, InstantiationException, IOException;

        @NonNegative
        public long getTimeToLive(String baseName, Locale locale);

        public boolean needsReload(@BinaryName String baseName, Locale locale, String format, ClassLoader loader, ResourceBundle bundle, long loadTime);

        @BinaryName
        public String toBundleName(@BinaryName String baseName, Locale locale);

        public final String toResourceName(String bundleName, String suffix);
    }

    private static class SingleFormatControl extends Control {

        protected SingleFormatControl(List<String> formats) {
        }

        public List<String> getFormats(String baseName);
    }

    private static final class NoFallbackControl extends SingleFormatControl {

        protected NoFallbackControl(List<String> formats) {
        }

        public Locale getFallbackLocale(String baseName, Locale locale);
    }

    private static class ResourceBundleProviderHelper {

        @SuppressWarnings("removal")
        static ResourceBundle newResourceBundle(Class<? extends ResourceBundle> bundleClass);

        static ResourceBundle loadResourceBundle(Module callerModule, Module module, String baseName, Locale locale);

        static boolean isAccessible(Module callerModule, Module module, String pn);

        static ResourceBundle loadPropertyResourceBundle(Module callerModule, Module module, String baseName, Locale locale) throws IOException;
    }
}
