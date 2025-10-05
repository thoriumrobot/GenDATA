/*
 * Copyright (c) 1995, 2021, Oracle and/or its affiliates. All rights reserved.
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

import org.checkerframework.checker.guieffect.qual.SafeEffect;
import org.checkerframework.checker.guieffect.qual.UIType;
import org.checkerframework.checker.nullness.qual.Nullable;
import org.checkerframework.checker.initialization.qual.UnknownInitialization;
import org.checkerframework.framework.qual.AnnotatedFor;
import java.awt.dnd.DropTarget;
import java.awt.event.AWTEventListener;
import java.awt.event.ComponentEvent;
import java.awt.event.ContainerEvent;
import java.awt.event.ContainerListener;
import java.awt.event.FocusEvent;
import java.awt.event.HierarchyEvent;
import java.awt.event.InputEvent;
import java.awt.event.KeyEvent;
import java.awt.event.MouseEvent;
import java.awt.event.MouseWheelEvent;
import java.awt.peer.ComponentPeer;
import java.awt.peer.ContainerPeer;
import java.awt.peer.LightweightPeer;
import java.beans.PropertyChangeListener;
import java.io.IOException;
import java.io.InvalidObjectException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.ObjectStreamField;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.Serial;
import java.io.Serializable;
import java.lang.ref.WeakReference;
import java.security.AccessController;
import java.util.ArrayList;
import java.util.EventListener;
import java.util.HashSet;
import java.util.Set;
import javax.accessibility.Accessible;
import javax.accessibility.AccessibleComponent;
import javax.accessibility.AccessibleContext;
import sun.awt.AWTAccessor;
import sun.awt.AWTAccessor.MouseEventAccessor;
import sun.awt.AppContext;
import sun.awt.PeerEvent;
import sun.awt.SunToolkit;
import sun.awt.dnd.SunDropTargetEvent;
import sun.java2d.pipe.Region;
import sun.security.action.GetBooleanAction;
import sun.util.logging.PlatformLogger;

@AnnotatedFor({ "guieffect", "nullness" })
@UIType
public class Container extends Component {

    public Container() {
    }

    @SuppressWarnings({ "unchecked", "rawtypes" })
    void initializeFocusTraversalKeys();

    public int getComponentCount();

    @Deprecated
    public int countComponents();

    public Component getComponent(int n);

    public Component[] getComponents();

    final Component[] getComponents_NoClientCode();

    Component[] getComponentsSync();

    public Insets getInsets();

    @Deprecated
    public Insets insets();

    public Component add(@UnknownInitialization(Container.class) Container this, Component comp);

    public Component add(@UnknownInitialization(Container.class) Container this, @Nullable String name, Component comp);

    public Component add(@UnknownInitialization(Container.class) Container this, Component comp, int index);

    boolean canContainFocusOwner(Component focusOwnerCandidate);

    final boolean hasHeavyweightDescendants();

    final boolean hasLightweightDescendants();

    Container getHeavyweightContainer();

    public void setComponentZOrder(Component comp, int index);

    public int getComponentZOrder(Component comp);

    public void add(@UnknownInitialization(Container.class) Container this, Component comp, @Nullable Object constraints);

    public void add(@UnknownInitialization(Container.class) Container this, Component comp, @Nullable Object constraints, int index);

    protected void addImpl(Component comp, @Nullable Object constraints, int index);

    @Override
    final boolean updateChildGraphicsData(GraphicsConfiguration gc);

    void checkGD(String stringID);

    public void remove(int index);

    public void remove(Component comp);

    public void removeAll();

    int numListening(long mask);

    void adjustListeningChildren(long mask, int num);

    void adjustDescendants(int num);

    void adjustDescendantsOnParent(int num);

    int countHierarchyMembers();

    final int createHierarchyEvents(int id, Component changed, Container changedParent, long changeFlags, boolean enabledOnToolkit);

    final void createChildHierarchyEvents(int id, long changeFlags, boolean enabledOnToolkit);

    @Nullable
    public LayoutManager getLayout();

    public void setLayout(@UnknownInitialization(Container.class) Container this, @Nullable LayoutManager mgr);

    public void doLayout();

    @Deprecated
    public void layout();

    public boolean isValidateRoot();

    @Override
    void invalidateParent();

    @SafeEffect
    @Override
    public void invalidate();

    public void validate();

    final void validateUnconditionally();

    protected void validateTree();

    void invalidateTree();

    public void setFont(@Nullable Font f);

    public Dimension getPreferredSize();

    @Deprecated
    public Dimension preferredSize();

    public Dimension getMinimumSize();

    @Deprecated
    public Dimension minimumSize();

    public Dimension getMaximumSize();

    public float getAlignmentX();

    public float getAlignmentY();

    public void paint(Graphics g);

    public void update(Graphics g);

    public void print(Graphics g);

    public void paintComponents(Graphics g);

    void lightweightPaint(Graphics g);

    void paintHeavyweightComponents(Graphics g);

    public void printComponents(Graphics g);

    void lightweightPrint(Graphics g);

    void printHeavyweightComponents(Graphics g);

    public synchronized void addContainerListener(ContainerListener l);

    public synchronized void removeContainerListener(ContainerListener l);

    public synchronized ContainerListener[] getContainerListeners();

    public <T extends EventListener> T[] getListeners(Class<T> listenerType);

    boolean eventEnabled(AWTEvent e);

    protected void processEvent(AWTEvent e);

    protected void processContainerEvent(ContainerEvent e);

    void dispatchEventImpl(AWTEvent e);

    void dispatchEventToSelf(AWTEvent e);

    Component getMouseEventTarget(int x, int y, boolean includeSelf);

    Component getDropTargetEventTarget(int x, int y, boolean includeSelf);

    static interface EventTargetFilter {

        boolean accept(final Component comp);
    }

    static class MouseEventTargetFilter implements EventTargetFilter {

        public boolean accept(final Component comp);
    }

    static class DropTargetEventTargetFilter implements EventTargetFilter {

        public boolean accept(final Component comp);
    }

    void proxyEnableEvents(long events);

    @Deprecated
    public void deliverEvent(Event e);

    @Nullable
    public Component getComponentAt(int x, int y);

    @Deprecated
    @Nullable
    public Component locate(int x, int y);

    @Nullable
    public Component getComponentAt(Point p);

    public Point getMousePosition(boolean allowChildren) throws HeadlessException;

    boolean isSameOrAncestorOf(Component comp, boolean allowChildren);

    public Component findComponentAt(int x, int y);

    final Component findComponentAt(int x, int y, boolean ignoreEnabled);

    final Component findComponentAtImpl(int x, int y, boolean ignoreEnabled);

    public Component findComponentAt(Point p);

    public void addNotify();

    public void removeNotify();

    public boolean isAncestorOf(Component c);

    static final class WakingRunnable implements Runnable {

        public void run();
    }

    protected String paramString();

    public void list(PrintStream out, int indent);

    public void list(PrintWriter out, int indent);

    public void setFocusTraversalKeys(int id, Set<? extends AWTKeyStroke> keystrokes);

    public Set<AWTKeyStroke> getFocusTraversalKeys(int id);

    public boolean areFocusTraversalKeysSet(int id);

    public boolean isFocusCycleRoot(Container container);

    final boolean containsFocus();

    void clearMostRecentFocusOwnerOnHide();

    void clearCurrentFocusCycleRootOnHide();

    final Container getTraversalRoot();

    public void setFocusTraversalPolicy(FocusTraversalPolicy policy);

    public FocusTraversalPolicy getFocusTraversalPolicy();

    public boolean isFocusTraversalPolicySet();

    public void setFocusCycleRoot(boolean focusCycleRoot);

    public boolean isFocusCycleRoot();

    public final void setFocusTraversalPolicyProvider(boolean provider);

    public final boolean isFocusTraversalPolicyProvider();

    public void transferFocusDownCycle();

    void preProcessKeyEvent(KeyEvent e);

    void postProcessKeyEvent(KeyEvent e);

    boolean postsOldMouseEvents();

    public void applyComponentOrientation(ComponentOrientation o);

    public void addPropertyChangeListener(PropertyChangeListener listener);

    public void addPropertyChangeListener(String propertyName, PropertyChangeListener listener);

    protected class AccessibleAWTContainer extends AccessibleAWTComponent {

        protected AccessibleAWTContainer() {
        }

        public int getAccessibleChildrenCount();

        public Accessible getAccessibleChild(int i);

        public Accessible getAccessibleAt(Point p);

        @SuppressWarnings("serial")
        protected ContainerListener accessibleContainerHandler;

        protected class AccessibleContainerHandler implements ContainerListener, Serializable {

            protected AccessibleContainerHandler() {
            }

            public void componentAdded(ContainerEvent e);

            public void componentRemoved(ContainerEvent e);
        }

        public void addPropertyChangeListener(PropertyChangeListener listener);

        public void removePropertyChangeListener(PropertyChangeListener listener);
    }

    Accessible getAccessibleAt(Point p);

    int getAccessibleChildrenCount();

    Accessible getAccessibleChild(int i);

    final void increaseComponentCount(Component c);

    final void decreaseComponentCount(Component c);

    @Override
    final Region getOpaqueShape();

    final void recursiveSubtractAndApplyShape(Region shape);

    final void recursiveSubtractAndApplyShape(Region shape, int fromZorder);

    final void recursiveSubtractAndApplyShape(Region shape, int fromZorder, int toZorder);

    final void recursiveApplyCurrentShape();

    final void recursiveApplyCurrentShape(int fromZorder);

    final void recursiveApplyCurrentShape(int fromZorder, int toZorder);

    final boolean isRecursivelyVisibleUpToHeavyweightContainer();

    @Override
    void mixOnShowing();

    @Override
    void mixOnHiding(boolean isLightweight);

    @Override
    void mixOnReshaping();

    @Override
    void mixOnZOrderChanging(int oldZorder, int newZorder);

    @Override
    void mixOnValidating();
}

class LightweightDispatcher implements java.io.Serializable, AWTEventListener {

    void dispose();

    void enableEvents(long events);

    boolean dispatchEvent(AWTEvent e);

    @SuppressWarnings("deprecation")
    public void eventDispatched(AWTEvent e);

    @SuppressWarnings("deprecation")
    void retargetMouseEvent(Component target, int id, MouseEvent e);
}
