/*
 * Copyright (c) 2000, 2021, Oracle and/or its affiliates. All rights reserved.
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
package java.awt;

import org.checkerframework.checker.interning.qual.UsesObjectEquals;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.event.FocusEvent;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.awt.event.WindowEvent;
import java.awt.peer.KeyboardFocusManagerPeer;
import java.awt.peer.LightweightPeer;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.beans.PropertyVetoException;
import java.beans.VetoableChangeListener;
import java.beans.VetoableChangeSupport;
import java.lang.ref.WeakReference;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.WeakHashMap;
import sun.util.logging.PlatformLogger;
import sun.awt.AppContext;
import sun.awt.SunToolkit;
import sun.awt.KeyboardFocusManagerPeerProvider;
import sun.awt.AWTAccessor;

@AnnotatedFor({ "interning" })
@UsesObjectEquals
public abstract class KeyboardFocusManager implements KeyEventDispatcher, KeyEventPostProcessor {

    public static final int FORWARD_TRAVERSAL_KEYS;

    public static final int BACKWARD_TRAVERSAL_KEYS;

    public static final int UP_CYCLE_TRAVERSAL_KEYS;

    public static final int DOWN_CYCLE_TRAVERSAL_KEYS;

    public static KeyboardFocusManager getCurrentKeyboardFocusManager();

    static synchronized KeyboardFocusManager getCurrentKeyboardFocusManager(AppContext appcontext);

    public static void setCurrentKeyboardFocusManager(KeyboardFocusManager newManager) throws SecurityException;

    final void setCurrentSequencedEvent(SequencedEvent current);

    final SequencedEvent getCurrentSequencedEvent();

    static Set<AWTKeyStroke> initFocusTraversalKeysSet(String value, Set<AWTKeyStroke> targetSet);

    public KeyboardFocusManager() {
    }

    public Component getFocusOwner();

    protected Component getGlobalFocusOwner() throws SecurityException;

    protected void setGlobalFocusOwner(Component focusOwner) throws SecurityException;

    public void clearFocusOwner();

    public void clearGlobalFocusOwner() throws SecurityException;

    @SuppressWarnings("removal")
    void clearGlobalFocusOwnerPriv();

    Component getNativeFocusOwner();

    void setNativeFocusOwner(Component comp);

    Window getNativeFocusedWindow();

    public Component getPermanentFocusOwner();

    protected Component getGlobalPermanentFocusOwner() throws SecurityException;

    protected void setGlobalPermanentFocusOwner(Component permanentFocusOwner) throws SecurityException;

    public Window getFocusedWindow();

    protected Window getGlobalFocusedWindow() throws SecurityException;

    protected void setGlobalFocusedWindow(Window focusedWindow) throws SecurityException;

    public Window getActiveWindow();

    protected Window getGlobalActiveWindow() throws SecurityException;

    protected void setGlobalActiveWindow(Window activeWindow) throws SecurityException;

    public synchronized FocusTraversalPolicy getDefaultFocusTraversalPolicy();

    public void setDefaultFocusTraversalPolicy(FocusTraversalPolicy defaultPolicy);

    public void setDefaultFocusTraversalKeys(int id, Set<? extends AWTKeyStroke> keystrokes);

    public Set<AWTKeyStroke> getDefaultFocusTraversalKeys(int id);

    public Container getCurrentFocusCycleRoot();

    protected Container getGlobalCurrentFocusCycleRoot() throws SecurityException;

    public void setGlobalCurrentFocusCycleRoot(Container newFocusCycleRoot) throws SecurityException;

    @SuppressWarnings("removal")
    void setGlobalCurrentFocusCycleRootPriv(final Container newFocusCycleRoot);

    public void addPropertyChangeListener(PropertyChangeListener listener);

    public void removePropertyChangeListener(PropertyChangeListener listener);

    public synchronized PropertyChangeListener[] getPropertyChangeListeners();

    public void addPropertyChangeListener(String propertyName, PropertyChangeListener listener);

    public void removePropertyChangeListener(String propertyName, PropertyChangeListener listener);

    public synchronized PropertyChangeListener[] getPropertyChangeListeners(String propertyName);

    protected void firePropertyChange(String propertyName, Object oldValue, Object newValue);

    public void addVetoableChangeListener(VetoableChangeListener listener);

    public void removeVetoableChangeListener(VetoableChangeListener listener);

    public synchronized VetoableChangeListener[] getVetoableChangeListeners();

    public void addVetoableChangeListener(String propertyName, VetoableChangeListener listener);

    public void removeVetoableChangeListener(String propertyName, VetoableChangeListener listener);

    public synchronized VetoableChangeListener[] getVetoableChangeListeners(String propertyName);

    protected void fireVetoableChange(String propertyName, Object oldValue, Object newValue) throws PropertyVetoException;

    public void addKeyEventDispatcher(KeyEventDispatcher dispatcher);

    public void removeKeyEventDispatcher(KeyEventDispatcher dispatcher);

    @SuppressWarnings("unchecked")
    protected synchronized java.util.List<KeyEventDispatcher> getKeyEventDispatchers();

    public void addKeyEventPostProcessor(KeyEventPostProcessor processor);

    public void removeKeyEventPostProcessor(KeyEventPostProcessor processor);

    @SuppressWarnings("unchecked")
    protected java.util.List<KeyEventPostProcessor> getKeyEventPostProcessors();

    static void setMostRecentFocusOwner(Component component);

    static synchronized void setMostRecentFocusOwner(Window window, Component component);

    static void clearMostRecentFocusOwner(Component comp);

    static synchronized Component getMostRecentFocusOwner(Window window);

    public abstract boolean dispatchEvent(AWTEvent e);

    public final void redispatchEvent(Component target, AWTEvent e);

    public abstract boolean dispatchKeyEvent(KeyEvent e);

    public abstract boolean postProcessKeyEvent(KeyEvent e);

    public abstract void processKeyEvent(Component focusedComponent, KeyEvent e);

    protected abstract void enqueueKeyEvents(long after, Component untilFocused);

    protected abstract void dequeueKeyEvents(long after, Component untilFocused);

    protected abstract void discardKeyEvents(Component comp);

    public abstract void focusNextComponent(Component aComponent);

    public abstract void focusPreviousComponent(Component aComponent);

    public abstract void upFocusCycle(Component aComponent);

    public abstract void downFocusCycle(Container aContainer);

    public final void focusNextComponent();

    public final void focusPreviousComponent();

    public final void upFocusCycle();

    public final void downFocusCycle();

    void dumpRequests();

    private static final class LightweightFocusRequest {

        public String toString();
    }

    private static final class HeavyweightFocusRequest {

        boolean addLightweightRequest(Component descendant, boolean temporary, FocusEvent.Cause cause);

        LightweightFocusRequest getFirstLightweightRequest();

        public String toString();
    }

    static boolean processSynchronousLightweightTransfer(Component heavyweight, Component descendant, boolean temporary, boolean focusedWindowChangeAllowed, long time);

    static int shouldNativelyFocusHeavyweight(Component heavyweight, Component descendant, boolean temporary, boolean focusedWindowChangeAllowed, long time, FocusEvent.Cause cause);

    static Window markClearGlobalFocusOwner();

    Component getCurrentWaitingRequest(Component parent);

    static boolean isAutoFocusTransferEnabled();

    static boolean isAutoFocusTransferEnabledFor(Component comp);

    static void processCurrentLightweightRequests();

    static FocusEvent retargetUnexpectedFocusEvent(FocusEvent fe);

    static FocusEvent retargetFocusGained(FocusEvent fe);

    static FocusEvent retargetFocusLost(FocusEvent fe);

    static AWTEvent retargetFocusEvent(AWTEvent event);

    void clearMarkers();

    static boolean removeFirstRequest();

    static void removeLastFocusRequest(Component heavyweight);

    static Component getHeavyweight(Component comp);

    static boolean isProxyActive(KeyEvent e);
}
