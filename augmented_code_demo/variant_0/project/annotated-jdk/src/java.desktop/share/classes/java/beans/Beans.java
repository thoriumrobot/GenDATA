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
package java.beans;

    @Positive
import org.checkerframework.checker.interning.qual.UsesObjectEquals;
    @Positive
import org.checkerframework.checker.nullness.qual.Nullable;
    @Positive
import org.checkerframework.framework.qual.AnnotatedFor;
    @Positive
import com.sun.beans.finder.ClassFinder;
    @Positive
import java.applet.Applet;
    @Positive
import java.applet.AppletContext;
    @Positive
import java.applet.AppletStub;
    @Positive
import java.applet.AudioClip;
    @Positive
import java.awt.Image;
    @Positive
import java.beans.beancontext.BeanContext;
    @Positive
import java.io.IOException;
    @Positive
import java.io.InputStream;
    @Positive
import java.io.ObjectInputStream;
    @Positive
import java.io.ObjectStreamClass;
    @Positive
import java.io.StreamCorruptedException;
    @Positive
import java.lang.reflect.Modifier;
    @Positive
import java.net.URL;
    @Positive
import java.util.Enumeration;
    @Positive
import java.util.Hashtable;
    @Positive
import java.util.Iterator;
    @Positive
import java.util.Vector;

    @Positive
@AnnotatedFor({ "interning", "nullness" })
    @Positive
@UsesObjectEquals
    @Positive
public class Beans {

    @Positive
    public Beans() {
    @Positive
    }

    @Positive
    public static Object instantiate(@Nullable ClassLoader cls, String beanName) throws IOException, ClassNotFoundException;

    @Positive
    @SuppressWarnings("deprecation")
    @Positive
    public static Object instantiate(@Nullable ClassLoader cls, String beanName, @Nullable BeanContext beanContext) throws IOException, ClassNotFoundException;

    @Positive
    @Deprecated()
    @Positive
    @SuppressWarnings("removal")
    @Positive
    public static Object instantiate(@Nullable ClassLoader cls, String beanName, @Nullable BeanContext beanContext, @Nullable AppletInitializer initializer) throws IOException, ClassNotFoundException;

    @Positive
    public static Object getInstanceOf(Object bean, Class<?> targetType);

    @Positive
    public static boolean isInstanceOf(Object bean, Class<?> targetType);

    @Positive
    public static boolean isDesignTime();

    @Positive
    public static boolean isGuiAvailable();

    @Positive
    public static void setDesignTime(boolean isDesignTime) throws SecurityException;

    @Positive
    public static void setGuiAvailable(boolean isGuiAvailable) throws SecurityException;
    @Positive
}

    @Positive
class ObjectInputStreamWithLoader extends ObjectInputStream {

    @Positive
    public ObjectInputStreamWithLoader(InputStream in, ClassLoader loader) throws IOException, StreamCorruptedException {
    @Positive
    }

    @Positive
    @SuppressWarnings("rawtypes")
    @Positive
    protected Class resolveClass(ObjectStreamClass classDesc) throws IOException, ClassNotFoundException;
    @Positive
}

    @Positive
@Deprecated()
    @Positive
@SuppressWarnings("removal")
    @Positive
class BeansAppletContext implements AppletContext {

    @Positive
    @Nullable
    @Positive
    public AudioClip getAudioClip(URL url);

    @Positive
    @Nullable
    @Positive
    public synchronized Image getImage(URL url);

    @Positive
    @Nullable
    @Positive
    public Applet getApplet(String name);

    @Positive
    public Enumeration<Applet> getApplets();

    @Positive
    public void showDocument(URL url);

    @Positive
    public void showDocument(URL url, String target);

    @Positive
    public void showStatus(String status);

    @Positive
    public void setStream(String key, InputStream stream) throws IOException;

    @Positive
    @Nullable
    @Positive
    public InputStream getStream(String key);

    @Positive
    @Nullable
    @Positive
    public Iterator<String> getStreamKeys();
    @Positive
}

    @Positive
@Deprecated()
    @Positive
@SuppressWarnings("removal")
    @Positive
class BeansAppletStub implements AppletStub {

    @Positive
    public boolean isActive();

    @Positive
    public URL getDocumentBase();

    @Positive
    public URL getCodeBase();

    @Positive
    @Nullable
    @Positive
    public String getParameter(String name);

    @Positive
    public AppletContext getAppletContext();

    @Positive
    public void appletResize(int width, int height);
    @Positive
}

// CFWR semantic augmentation - variant 0
