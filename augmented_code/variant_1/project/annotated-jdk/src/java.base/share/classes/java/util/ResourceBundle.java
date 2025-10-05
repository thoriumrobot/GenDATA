/*
 * CFWR semantic augmentation: applied semantic-preserving transformations.
 */
/*
    @Positive
 * Copyright (c) 1996, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.util;

    @Positive
import org.checkerframework.checker.i18n.qual.LocalizableKey;
    @Positive
import org.checkerframework.checker.i18n.qual.Localized;
    @Positive
import org.checkerframework.checker.i18nformatter.qual.I18nMakeFormat;
    @Positive
import org.checkerframework.checker.index.qual.NonNegative;
    @Positive
import org.checkerframework.checker.lock.qual.GuardSatisfied;
    @Positive
import org.checkerframework.checker.nullness.qual.EnsuresKeyForIf;
    @Positive
import org.checkerframework.checker.nullness.qual.KeyFor;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.checker.propkey.qual.PropertyKey;
    @Positive
import org.checkerframework.checker.signature.qual.BinaryName;
    @Positive
import org.checkerframework.dataflow.qual.Pure;
    @Positive
import org.checkerframework.dataflow.qual.SideEffectFree;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import org.checkerframework.framework.qual.CFComment;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.UncheckedIOException;
    @Positive
import java.lang.ref.Reference;
    @Positive
import java.lang.ref.ReferenceQueue;
    @Positive
import java.lang.ref.SoftReference;
    @Positive
import java.lang.ref.WeakReference;
    @Positive
import java.lang.reflect.Constructor;
    @Positive
import java.lang.reflect.InvocationTargetException;
    @Positive
import java.lang.reflect.Modifier;
    @Positive
import java.net.JarURLConnection;
    @Positive
import java.net.URL;
    @Positive
import java.net.URLConnection;
    @Positive
import java.security.AccessController;
    @Positive
import java.security.PrivilegedAction;
    @Positive
import java.security.PrivilegedActionException;
    @Positive
import java.security.PrivilegedExceptionAction;
    @Positive
import java.util.concurrent.ConcurrentHashMap;
    @Positive
import java.util.concurrent.ConcurrentMap;
    @Positive
import java.util.jar.JarEntry;
    @Positive
import java.util.spi.ResourceBundleControlProvider;
    @Positive
import java.util.spi.ResourceBundleProvider;
    @Positive
import java.util.stream.Stream;
    @Positive
import jdk.internal.loader.BootLoader;
    @Positive
import jdk.internal.access.JavaUtilResourceBundleAccess;
    @Positive
import jdk.internal.access.SharedSecrets;
    @Positive
import jdk.internal.reflect.CallerSensitive;
    @Positive
import jdk.internal.reflect.Reflection;
    @Positive
import sun.security.action.GetPropertyAction;
    @Positive
import sun.util.locale.BaseLocale;
    @Positive
import sun.util.locale.LocaleObjectCache;
    @Positive
import sun.util.resources.Bundles;
    @Positive
import static sun.security.util.SecurityConstants.GET_CLASSLOADER_PERMISSION;

    @Positive
@AnnotatedFor({ "i18n", "i18nformatter", "index", "lock", "nullness", "propkey", "signature" })
    @Positive
public abstract class ResourceBundle {

    @Positive
    public String getBaseBundleName();

    @Positive
    protected ResourceBundle parent;

    @Positive
    public ResourceBundle() {
    @Positive
    }

    @Positive
    @I18nMakeFormat
    @Positive
    @Localized
    @Positive
    public final String getString(@LocalizableKey @PropertyKey String key);

    @Positive
    @Localized
    @Positive
    public final String[] getStringArray(@LocalizableKey @PropertyKey String key);

    @Positive
    @Localized
    @Positive
    public final Object getObject(@LocalizableKey @PropertyKey String key);

    @Positive
    public Locale getLocale();

    @Positive
    protected void setParent(ResourceBundle parent);

    @Positive
    private static final class CacheKey {

    @Positive
        String getName();

    @Positive
        Locale getLocale();

    @Positive
        CacheKey setLocale(Locale locale);

    @Positive
        Module getModule();

    @Positive
        Module getCallerModule();

    @Positive
        ServiceLoader<ResourceBundleProvider> getProviders();

    @Positive
        boolean hasProviders();

    @Positive
        boolean callerHasProvider();

    @Positive
        @Override
    @Positive
        public boolean equals(Object other);

    @Positive
        @Override
    @Positive
        public int hashCode();

    @Positive
        String getFormat();

    @Positive
        void setFormat(String format);

    @Positive
        @Override
    @Positive
        public String toString();
    @Positive
    }

    @Positive
    private static interface CacheKeyReference {

    @Positive
        public CacheKey getCacheKey();
    @Positive
    }

    @Positive
    private static class KeyElementReference<T> extends WeakReference<T> implements CacheKeyReference {

    @Positive
        @Override
    @Positive
        public CacheKey getCacheKey();
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
    @CallerSensitive
    @Positive
    public static final ResourceBundle getBundle(@BinaryName String baseName);

    @Positive
    @CallerSensitive
    @Positive
    public static final ResourceBundle getBundle(@BinaryName String baseName, Control control);

    @Positive
    @CallerSensitive
    @Positive
    public static final ResourceBundle getBundle(@BinaryName String baseName, Locale locale);

    @Positive
    @CallerSensitive
    @Positive
    public static ResourceBundle getBundle(@BinaryName String baseName, Module module);

    @Positive
    @CallerSensitive
    @Positive
    public static ResourceBundle getBundle(@BinaryName String baseName, Locale targetLocale, Module module);

    @Positive
    @CallerSensitive
    @Positive
    public static final ResourceBundle getBundle(@BinaryName String baseName, Locale targetLocale, Control control);

    @Positive
    @CallerSensitive
    @Positive
    public static ResourceBundle getBundle(@BinaryName String baseName, Locale locale, ClassLoader loader);

    @Positive
    @CallerSensitive
    @Positive
    public static ResourceBundle getBundle(@BinaryName String baseName, Locale targetLocale, ClassLoader loader, Control control);

    @Positive
    private static class ResourceBundleControlProviderHolder {
    @Positive
    }

    @Positive
    @CallerSensitive
    @Positive
    public static final void clearCache();

    @Positive
    public static final void clearCache(ClassLoader loader);

    @Positive
    protected abstract Object handleGetObject(String key);

    @Positive
    @SideEffectFree
    @Positive
    public abstract Enumeration<String> getKeys(@GuardSatisfied ResourceBundle this);

    @Positive
    @Pure
    @Positive
    @EnsuresKeyForIf(result = true, expression = "#1", map = "this")
    @Positive
    public boolean containsKey(@GuardSatisfied ResourceBundle this, String key);

    @Positive
    @SideEffectFree
    @Positive
    public Set<@KeyFor("this") @LocalizableKey @PropertyKey String> keySet(@GuardSatisfied ResourceBundle this);

    @Positive
    protected Set<String> handleKeySet();

    @Positive
    public static class Control {

    @Positive
        public static final List<String> FORMAT_DEFAULT;

    @Positive
        public static final List<String> FORMAT_CLASS;

    @Positive
        public static final List<String> FORMAT_PROPERTIES;

    @Positive
        public static final long TTL_DONT_CACHE;

    @Positive
        public static final long TTL_NO_EXPIRATION_CONTROL;

    @Positive
        protected Control() {
    @Positive
        }

    @Positive
        public static final Control getControl(List<String> formats);

    @Positive
        public static final Control getNoFallbackControl(List<String> formats);

    @Positive
        public List<String> getFormats(String baseName);

    @Positive
        public List<Locale> getCandidateLocales(String baseName, Locale locale);

    @Positive
        private static class CandidateListCache extends LocaleObjectCache<BaseLocale, List<Locale>> {

    @Positive
            protected List<Locale> createObject(BaseLocale base);
    @Positive
        }

    @Positive
        public Locale getFallbackLocale(String baseName, Locale locale);

    @Positive
        public ResourceBundle newBundle(@BinaryName String baseName, Locale locale, String format, ClassLoader loader, boolean reload) throws IllegalAccessException, InstantiationException, IOException;

    @Positive
        @NonNegative
    @Positive
        public long getTimeToLive(String baseName, Locale locale);

    @Positive
        public boolean needsReload(@BinaryName String baseName, Locale locale, String format, ClassLoader loader, ResourceBundle bundle, long loadTime);

    @Positive
        @BinaryName
    @Positive
        public String toBundleName(@BinaryName String baseName, Locale locale);

    @Positive
        public final String toResourceName(String bundleName, String suffix);
    @Positive
    }

    @Positive
    private static class SingleFormatControl extends Control {

    @Positive
        protected SingleFormatControl(List<String> formats) {
    @Positive
        }

    @Positive
        public List<String> getFormats(String baseName);
    @Positive
    }

    @Positive
    private static final class NoFallbackControl extends SingleFormatControl {

    @Positive
        protected NoFallbackControl(List<String> formats) {
    @Positive
        }

    @Positive
        public Locale getFallbackLocale(String baseName, Locale locale);
    @Positive
    }

    @Positive
    private static class ResourceBundleProviderHelper {

    @Positive
        @SuppressWarnings("removal")
    @Positive
        static ResourceBundle newResourceBundle(Class<? extends ResourceBundle> bundleClass);

    @Positive
        static ResourceBundle loadResourceBundle(Module callerModule, Module module, String baseName, Locale locale);

    @Positive
        static boolean isAccessible(Module callerModule, Module module, String pn);

    @Positive
        static ResourceBundle loadPropertyResourceBundle(Module callerModule, Module module, String baseName, Locale locale) throws IOException;
    @Positive
    }
    @Positive
}
