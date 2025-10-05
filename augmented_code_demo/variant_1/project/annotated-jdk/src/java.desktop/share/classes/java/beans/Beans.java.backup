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
package java.beans;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.framework.qual.AnnotatedFor;
import com.sun.beans.finder.ClassFinder;
import java.applet.Applet;
import java.applet.AppletContext;
import java.applet.AppletStub;
import java.applet.AudioClip;
import java.awt.Image;
import java.beans.beancontext.BeanContext;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectStreamClass;
import java.io.StreamCorruptedException;
import java.lang.reflect.Modifier;
import java.net.URL;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.Vector;

@AnnotatedFor({ "interning", "nullness" })
@UsesObjectEquals
public class Beans {

    public Beans() {
    }

    public static Object instantiate(@Nullable ClassLoader cls, String beanName) throws IOException, ClassNotFoundException;

    @SuppressWarnings("deprecation")
    public static Object instantiate(@Nullable ClassLoader cls, String beanName, @Nullable BeanContext beanContext) throws IOException, ClassNotFoundException;

    @Deprecated()
    @SuppressWarnings("removal")
    public static Object instantiate(@Nullable ClassLoader cls, String beanName, @Nullable BeanContext beanContext, @Nullable AppletInitializer initializer) throws IOException, ClassNotFoundException;

    public static Object getInstanceOf(Object bean, Class<?> targetType);

    public static boolean isInstanceOf(Object bean, Class<?> targetType);

    public static boolean isDesignTime();

    public static boolean isGuiAvailable();

    public static void setDesignTime(boolean isDesignTime) throws SecurityException;

    public static void setGuiAvailable(boolean isGuiAvailable) throws SecurityException;
}

class ObjectInputStreamWithLoader extends ObjectInputStream {

    public ObjectInputStreamWithLoader(InputStream in, ClassLoader loader) throws IOException, StreamCorruptedException {
    }

    @SuppressWarnings("rawtypes")
    protected Class resolveClass(ObjectStreamClass classDesc) throws IOException, ClassNotFoundException;
}

@Deprecated()
@SuppressWarnings("removal")
class BeansAppletContext implements AppletContext {

    @Nullable
    public AudioClip getAudioClip(URL url);

    @Nullable
    public synchronized Image getImage(URL url);

    @Nullable
    public Applet getApplet(String name);

    public Enumeration<Applet> getApplets();

    public void showDocument(URL url);

    public void showDocument(URL url, String target);

    public void showStatus(String status);

    public void setStream(String key, InputStream stream) throws IOException;

    @Nullable
    public InputStream getStream(String key);

    @Nullable
    public Iterator<String> getStreamKeys();
}

@Deprecated()
@SuppressWarnings("removal")
class BeansAppletStub implements AppletStub {

    public boolean isActive();

    public URL getDocumentBase();

    public URL getCodeBase();

    @Nullable
    public String getParameter(String name);

    public AppletContext getAppletContext();

    public void appletResize(int width, int height);
}
